"""
Claude-Assisted Script

Took Simon's SSVEP script and asked Claude to replace the Speller with the Go/NoGo game. 

Go/No-Go EEG Data Collection Script
=====================================
Adapted from run_vep.py (32-class SSVEP speller).
Hardware: OpenBCI Cyton 8-channel via BrainFlow.

Task Design:
    - Go trials  (80%): Press SPACE when you see a GREEN circle.
    - No-Go trials (20%): Withhold response when you see a RED circle.

Trial outcomes saved per trial:
    - 'hit'               : Go trial, correct response
    - 'miss'              : Go trial, no response (omission error)
    - 'commission_error'  : No-Go trial, incorrect response <- KEY TRIALS
    - 'correct_rejection' : No-Go trial, correctly withheld

EEG epochs are saved time-locked to the BUTTON PRESS (not stimulus onset)
so that the pre-response window (-500 ms to 0 ms) can be extracted offline
for error-vs-correct classification.

Data saved per run:
    eeg_raw.npy          - continuous raw EEG (8 ch x N samples)
    aux_raw.npy          - continuous aux/analog channels (photosensor on ch1)
    trial_metadata.npy   - list of dicts with per-trial info
    eeg_trials.npy       - epoched EEG (trials x ch x samples), response-locked
    labels.npy           - per-trial outcome string
"""

# ---------------------------------------------
#  CONFIGURATION  -- edit these before recording
# ---------------------------------------------
SUBJECT          = 1          # Subject number
SESSION          = 1          # Session number
RUN              = 1          # Run number (also used as random seed)
CYTON_IN         = False      # True = record from OpenBCI Cyton; False = dry run
CYTON_BOARD_ID   = 0          # 0 = Cyton (8ch), 2 = Cyton+Daisy (16ch)
SAMPLING_RATE    = 250        # Hz -- must match board setting

# Display -- Mac 
SCREEN_WIDTH     = 1440       # actual screen resolution (not scaled)
SCREEN_HEIGHT    = 900
REFRESH_RATE     = 60.0       # Hz -- adjust to your monitor

# Display -- Windows
# SCREEN_WIDTH  = 1920  # change to match the Windows PC screen
# SCREEN_HEIGHT = 1080
# REFRESH_RATE  = 60.0  # confirm in Windows Display Settings

# Task timing (all in seconds)
FIXATION_DURATION  = 0.5      # Duration of fixation cross
STIM_DURATION      = 0.25     # How long the Go/No-Go circle is shown
RESPONSE_WINDOW    = 0.6      # Time after stimulus onset to accept a response
ITI_MIN            = 0.8      # Minimum inter-trial interval
ITI_MAX            = 1.2      # Maximum inter-trial interval (jittered)

# Trial counts
N_GO_TRIALS        = 60      # Number of Go trials per run
N_NOGO_TRIALS      = 15       # Number of No-Go trials (20% of total = 30/150)

# Epoch window around button press (for offline pre-response analysis)
EPOCH_PRE_PRESS    = 0.7      # seconds BEFORE press (gives -700 ms buffer; use -500ms offline)
EPOCH_POST_PRESS   = 0.2      # seconds AFTER press

# File paths
SAVE_DIR = f'data/gonogo/sub-{SUBJECT:02d}/ses-{SESSION:02d}/'

# ---------------------------------------------
#  IMPORTS
# ---------------------------------------------
import os, random, pickle, time, platform
import numpy as np
from psychopy import visual, core, event
from psychopy.hardware import keyboard as kb
import mne

# ---------------------------------------------
#  CYTON / BRAINFLOW SETUP
# ---------------------------------------------
if CYTON_IN:
    import glob, sys, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue

    BAUD_RATE     = 115200
    ANALOGUE_MODE = '/2'   # Enables analog reads on A5/A6/A7 -- photosensor on A6 (aux ch 1)

    def find_openbci_port():
        """Auto-detect the serial port of the OpenBCI Cyton dongle."""
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Unsupported OS for port detection.')
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    while '$$$' not in line:
                        line += s.read().decode('utf-8', errors='replace')
                    if 'OpenBCI' in line:
                        s.close()
                        return port
                s.close()
            except (OSError, serial.SerialException):
                pass
        raise OSError('Cannot find OpenBCI Cyton dongle. Check USB connection.')

    print('[CYTON] Connecting...')
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    else:
        params.ip_port = 9000
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)
    print('[CYTON] Stream started.')

    stop_event = Event()
    queue_in   = Queue()

    def _get_data_thread(q):
        """Background thread: polls board and pushes chunks to queue."""
        while not stop_event.is_set():
            data = board.get_board_data()
            ts  = data[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg = data[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux = data[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(ts) > 0:
                q.put((eeg, aux, ts))
            time.sleep(0.05)

    cyton_thread = Thread(target=_get_data_thread, args=(queue_in,), daemon=True)
    cyton_thread.start()

    # Running buffers -- data accumulates here throughout the session
    eeg_buf = np.zeros((8, 0))
    aux_buf = np.zeros((3, 0))
    ts_buf  = np.zeros((0,))


def drain_queue():
    """Flush all pending data from the background thread into the running buffers."""
    global eeg_buf, aux_buf, ts_buf
    while not queue_in.empty():
        eeg_in, aux_in, ts_in = queue_in.get()
        eeg_buf = np.concatenate((eeg_buf, eeg_in), axis=1)
        aux_buf = np.concatenate((aux_buf, aux_in), axis=1)
        ts_buf  = np.concatenate((ts_buf, ts_in),  axis=0)


def current_sample_index():
    """Return the latest sample index in the buffer (used to timestamp button presses)."""
    drain_queue()
    return eeg_buf.shape[1]


def save_all_data(trial_metadata, eeg_trials_list, labels_list):
    """Save all data to disk."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    run_tag = f'run-{RUN:02d}'
    np.save(SAVE_DIR + f'eeg_raw_{run_tag}.npy',        eeg_buf)
    np.save(SAVE_DIR + f'aux_raw_{run_tag}.npy',        aux_buf)
    np.save(SAVE_DIR + f'trial_metadata_{run_tag}.npy', np.array(trial_metadata, dtype=object))
    np.save(SAVE_DIR + f'eeg_trials_{run_tag}.npy',     np.array(eeg_trials_list))
    np.save(SAVE_DIR + f'labels_{run_tag}.npy',         np.array(labels_list))
    print(f'[SAVE] Data saved to {SAVE_DIR}')
    print(f'       EEG buffer shape : {eeg_buf.shape}')
    print(f'       Trials saved     : {len(eeg_trials_list)}')
    print(f'       Labels           : {np.unique(labels_list, return_counts=True)}')


# ---------------------------------------------
#  PSYCHOPY WINDOW  (macOS / Windows compatible)
# ---------------------------------------------
IS_MAC = platform.system() == 'Darwin'

win = visual.Window(
    size=[SCREEN_WIDTH, SCREEN_HEIGHT],
    fullscr=True,
    allowGUI=True,          # must be True on macOS to receive keyboard events
    color=[-0.6, -0.6, -0.6],
    checkTiming=True,
    useRetina=True,         # required on all Retina Macs; ignored on non-Retina
)

# macOS: bring window to front so keyboard events route here, not to terminal
if IS_MAC:
    try:
        win.winHandle.activate()
        win.winHandle.set_mouse_visible(True)
    except Exception:
        pass

# 'event' backend is reliable on all platforms when using a frame-driven
# getKeys() loop (which we do). iohub requires psutil + a background server
# and is unnecessary complexity for this setup.
keyboard = kb.Keyboard(backend='event')

ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT

# ---------------------------------------------
#  STIMULI DEFINITIONS
# ---------------------------------------------
fixation = visual.TextStim(win, text='+', height=0.12, color='white', units='norm')

go_stim = visual.Circle(
    win, radius=0.18, fillColor=[0.0, 0.85, 0.2],
    lineColor=None, units='norm'
)

nogo_stim = visual.Circle(
    win, radius=0.18, fillColor=[0.9, 0.1, 0.1],
    lineColor=None, units='norm'
)

instruction_text = visual.TextStim(
    win, wrapWidth=1.6, height=0.07, color='white', units='norm',
    text=(
        "GO / NO-GO TASK\n\n"
        "GREEN circle  ->  Press SPACE as fast as you can\n"
        "RED circle    ->  Do NOT press anything\n\n"
        "Try to respond quickly but accurately.\n\n"
        "Press SPACE to begin."
    )
)

trial_counter = visual.TextStim(
    win, pos=(0, -0.92), height=0.055, color='grey', units='norm', text=''
)

feedback_text = visual.TextStim(win, pos=(0, -0.7), height=0.07, units='norm', text='')

photosensor = visual.Rect(
    win, units='norm',
    width=0.08, height=0.08 * ASPECT,
    pos=[1 - 0.04, -1 + 0.04 * ASPECT],
    fillColor='black', lineWidth=0
)


def show_photosensor(state):
    photosensor.fillColor = 'white' if state else 'black'


def draw_background():
    photosensor.draw()
    trial_counter.draw()


# ---------------------------------------------
#  TRIAL SEQUENCE
# ---------------------------------------------
def build_trial_sequence(n_go, n_nogo, seed=0):
    rng = np.random.default_rng(seed)
    trials = [{'trial_type': 'go'}] * n_go + [{'trial_type': 'nogo'}] * n_nogo
    rng.shuffle(trials)
    for t in trials:
        t['iti'] = rng.uniform(ITI_MIN, ITI_MAX)
    return trials


# ---------------------------------------------
#  EPOCH EXTRACTION (response-locked)
# ---------------------------------------------
def extract_response_locked_epoch(press_sample):
    """
    Extract a pre-response-locked EEG epoch centred on the button press.
    Returns epoch of shape (8 channels, n_samples) or None if buffer too short.

    Critical analysis window offline: -500 ms to 0 ms before press.
    We save -EPOCH_PRE_PRESS to +EPOCH_POST_PRESS so you can choose the
    exact window during analysis.
    """
    pre_samples  = int(EPOCH_PRE_PRESS  * SAMPLING_RATE)
    post_samples = int(EPOCH_POST_PRESS * SAMPLING_RATE)
    start = press_sample - pre_samples
    end   = press_sample + post_samples

    if start < 0 or end > eeg_buf.shape[1]:
        return None

    epoch_raw = eeg_buf[:, start:end].copy()
    epoch_filt = mne.filter.filter_data(
        epoch_raw, sfreq=SAMPLING_RATE, l_freq=1.0, h_freq=40.0, verbose=False
    )
    # Baseline correction: mean of first 200 ms
    baseline_end = int(0.2 * SAMPLING_RATE)
    baseline_mean = np.mean(epoch_filt[:, :baseline_end], axis=1, keepdims=True)
    epoch_filt -= baseline_mean
    return epoch_filt


# ---------------------------------------------
#  MAIN EXPERIMENT LOOP
# ---------------------------------------------
trial_metadata  = []
eeg_trials_list = []
labels_list     = []

# Blank flip to force macOS focus transfer before first waitKeys
win.flip()
core.wait(0.1)

instruction_text.draw()
win.flip()
keys = event.waitKeys(keyList=['space', 'escape'])
if 'escape' in keys:
    win.close()
    core.quit()

trial_sequence = build_trial_sequence(N_GO_TRIALS, N_NOGO_TRIALS, seed=RUN)
n_trials = len(trial_sequence)

print(f'[TASK] Starting run {RUN}: {N_GO_TRIALS} Go + {N_NOGO_TRIALS} No-Go = {n_trials} trials')

for i_trial, trial in enumerate(trial_sequence):
    trial_type  = trial['trial_type']
    iti         = trial['iti']
    is_go       = (trial_type == 'go')

    trial_counter.text = f'{i_trial + 1} / {n_trials}'

    # -- ITI: fixation cross --
    fixation.draw()
    show_photosensor(False)
    draw_background()
    win.flip()
    core.wait(iti)

    # -- Stimulus onset --
    stim = go_stim if is_go else nogo_stim
    event.clearEvents()   # flush stale keypresses before stimulus onset

    stim.draw()
    show_photosensor(True)
    draw_background()
    win.flip()
    stim_onset_sample = current_sample_index() if CYTON_IN else None

    # -- Response window (frame-by-frame loop) --
    # Driving with win.flip() every ~16 ms pumps the OS event queue continuously.
    # getKeys() after each flip catches every press with <16 ms latency.
    #
    # Timeline:
    #   0 ms       -- stimulus appears, listening begins
    #   0-250 ms   -- stimulus visible
    #   250 ms     -- stimulus disappears, still listening
    #   600 ms     -- response window closes
    #
    response_rt  = None
    press_sample = None
    rt_clock     = core.Clock()   # ticks from stimulus onset flip

    n_frames_total = int(RESPONSE_WINDOW * REFRESH_RATE)
    n_frames_stim  = int(STIM_DURATION   * REFRESH_RATE)

    for i_frame in range(n_frames_total):
        if i_frame == n_frames_stim:
            # First blank frame -- hide stimulus
            show_photosensor(False)
            draw_background()
            win.flip()
        elif i_frame > n_frames_stim:
            # Remaining blank frames -- just flip to keep event queue alive
            draw_background()
            win.flip()
        else:
            # Stimulus still visible
            stim.draw()
            show_photosensor(True)
            draw_background()
            win.flip()

        # Check keyboard after every flip
        keys = event.getKeys(keyList=['space', 'escape'], timeStamped=rt_clock)
        for key_name, key_rt in keys:
            if key_name == 'escape':
                if CYTON_IN:
                    save_all_data(trial_metadata, eeg_trials_list, labels_list)
                    stop_event.set()
                    board.stop_stream()
                    board.release_session()
                win.close()
                core.quit()
            if key_name == 'space' and response_rt is None:
                response_rt  = key_rt
                press_sample = stim_onset_sample + int(response_rt * SAMPLING_RATE) if CYTON_IN else None

        if response_rt is not None:
            break

    # -- Classify outcome --
    if is_go:
        outcome = 'hit' if response_rt is not None else 'miss'
    else:
        outcome = 'commission_error' if response_rt is not None else 'correct_rejection'

    # -- Extract EEG epoch (response-locked) --
    epoch = None
    if CYTON_IN and press_sample is not None:
        drain_queue()
        needed = press_sample + int(EPOCH_POST_PRESS * SAMPLING_RATE)
        wait_attempts = 0
        while eeg_buf.shape[1] < needed and wait_attempts < 20:
            drain_queue()
            time.sleep(0.05)
            wait_attempts += 1
        epoch = extract_response_locked_epoch(press_sample)

    meta = {
        'trial_idx'         : i_trial,
        'trial_type'        : trial_type,
        'outcome'           : outcome,
        'rt_s'              : response_rt,
        'stim_onset_sample' : stim_onset_sample,
        'press_sample'      : press_sample,
        'epoch_pre_s'       : EPOCH_PRE_PRESS,
        'epoch_post_s'      : EPOCH_POST_PRESS,
    }
    trial_metadata.append(meta)

    if epoch is not None:
        eeg_trials_list.append(epoch)
        labels_list.append(outcome)

    # Feedback display (set SHOW_FEEDBACK = False for real EEG recording)
    SHOW_FEEDBACK = True
    if SHOW_FEEDBACK:
        color_map = {
            'hit'              : [0.0, 0.8, 0.2],
            'miss'             : [0.9, 0.5, 0.0],
            'commission_error' : [0.9, 0.1, 0.1],
            'correct_rejection': [0.5, 0.5, 0.9],
        }
        feedback_text.text  = outcome.replace('_', ' ').upper()
        feedback_text.color = color_map[outcome]
        feedback_text.draw()
        show_photosensor(False)
        draw_background()
        win.flip()
        core.wait(0.25)

    rt_str = f'{response_rt*1000:.0f} ms' if response_rt else '---'
    print(f'Trial {i_trial+1:3d}/{n_trials}  [{trial_type:5s}]  {outcome:<22s}  RT: {rt_str}')


# ---------------------------------------------
#  END OF RUN
# ---------------------------------------------
end_text = visual.TextStim(
    win, height=0.08, color='white', units='norm',
    text="Run complete!\n\nSaving data...\n\nPlease wait."
)
end_text.draw()
win.flip()

if CYTON_IN:
    save_all_data(trial_metadata, eeg_trials_list, labels_list)
    stop_event.set()
    board.stop_stream()
    board.release_session()
else:
    outcomes = [m['outcome'] for m in trial_metadata]
    for outcome in ['hit', 'miss', 'commission_error', 'correct_rejection']:
        print(f'  {outcome:<22s}: {outcomes.count(outcome)}')

hits    = sum(1 for m in trial_metadata if m['outcome'] == 'hit')
errors  = sum(1 for m in trial_metadata if m['outcome'] == 'commission_error')
misses  = sum(1 for m in trial_metadata if m['outcome'] == 'miss')
go_acc  = hits / N_GO_TRIALS * 100
nogo_er = errors / N_NOGO_TRIALS * 100

summary = visual.TextStim(
    win, height=0.07, color='white', units='norm',
    text=(
        f"Run {RUN} Summary\n\n"
        f"Go accuracy     : {go_acc:.1f}%  ({hits}/{N_GO_TRIALS})\n"
        f"No-Go errors    : {nogo_er:.1f}%  ({errors}/{N_NOGO_TRIALS})\n"
        f"Omission misses : {misses}\n\n"
        f"Data saved to:\n{SAVE_DIR}\n\n"
        f"Press SPACE to exit."
    )
)
summary.draw()
win.flip()
event.waitKeys(keyList=['space'])

win.close()
core.quit()
