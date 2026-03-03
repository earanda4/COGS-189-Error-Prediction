"""
Flanker Task EEG Data Collection Script (Non-Blocking Fix)
==========================================================
Hardware: OpenBCI Cyton 8-channel via BrainFlow.
Platform: Mac / Windows compatible.

Task Design:
    - Stimulus: A row of 5 arrows (e.g., <<<<< or >><>>)
    - Task: Press key corresponding to the CENTER arrow.
        - Left Center (<)  -> Press 'F'
        - Right Center (>) -> Press 'J'
    - Timing:
        - Stimulus appears for 0.2s
        - Blank screen for remainder of 1.0s window
        - You can press the key at ANY time during this 1.0s window.
"""

# ---------------------------------------------
#  CONFIGURATION
# ---------------------------------------------
SUBJECT          = 1          # Subject number
SESSION          = 1          # Session number
RUN              = 1          # Run number
CYTON_IN         = True       # True = record; False = dry run
CYTON_BOARD_ID   = 0          # 0 = Cyton, 2 = Cyton+Daisy
SAMPLING_RATE    = 250        # Hz

# Display
SCREEN_WIDTH     = 1536
SCREEN_HEIGHT    = 864
REFRESH_RATE     = 60.0

# Task timing (seconds)
STIM_DURATION      = 0.2      # How long arrow is visible
RESPONSE_WINDOW    = 1.0      # Total time to respond (Stim + Blank)
ITI_MIN            = 0.8      # Random pause between trials
ITI_MAX            = 1.2

# Trial counts
N_CONGRUENT        = 75       # 50%
N_INCONGRUENT      = 75       # 50% 

# Epoch window (Response-Locked)
EPOCH_PRE_PRESS    = 0.7      
EPOCH_POST_PRESS   = 0.2     

# File paths
SAVE_DIR = f'data/flanker/sub-{SUBJECT:02d}/ses-{SESSION:02d}/'

# ---------------------------------------------
#  IMPORTS
# ---------------------------------------------
import os, random, time, sys, glob
import numpy as np
from psychopy import visual, core, event
from psychopy.hardware import keyboard as kb
import mne

# ---------------------------------------------
#  CYTON / BRAINFLOW SETUP
# ---------------------------------------------
if CYTON_IN:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue

    BAUD_RATE    = 115200
    ANALOGUE_MODE = '/2'

    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        elif sys.platform.startswith('linux'):
            ports = glob.glob('/dev/ttyUSB*')
        else:
            raise EnvironmentError('Unsupported OS.')
        
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                time.sleep(2)
                if s.inWaiting():
                    if 'OpenBCI' in s.read(s.inWaiting()).decode('utf-8', errors='ignore'):
                        s.close()
                        return port
                s.close()
            except:
                pass
        raise OSError('Cannot find OpenBCI Cyton.')

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
        while not stop_event.is_set():
            data = board.get_board_data()
            if data.size > 0:
                ts  = data[board.get_timestamp_channel(CYTON_BOARD_ID)]
                eeg = data[board.get_eeg_channels(CYTON_BOARD_ID)]
                aux = data[board.get_analog_channels(CYTON_BOARD_ID)]
                q.put((eeg, aux, ts))
            time.sleep(0.005) # Fast polling

    cyton_thread = Thread(target=_get_data_thread, args=(queue_in,), daemon=True)
    cyton_thread.start()

    eeg_buf = np.zeros((8, 0))
    aux_buf = np.zeros((3, 0))
    ts_buf  = np.zeros((0,))

def drain_queue():
    global eeg_buf, aux_buf, ts_buf
    while CYTON_IN and not queue_in.empty():
        eeg_in, aux_in, ts_in = queue_in.get()
        eeg_buf = np.concatenate((eeg_buf, eeg_in), axis=1)
        aux_buf = np.concatenate((aux_buf, aux_in), axis=1)
        ts_buf  = np.concatenate((ts_buf, ts_in),  axis=0)

def current_sample_index():
    drain_queue()
    return eeg_buf.shape[1] if CYTON_IN else 0

def save_all_data(trial_metadata, eeg_trials_list, labels_list):
    os.makedirs(SAVE_DIR, exist_ok=True)
    run_tag = f'run-{RUN:02d}'
    if CYTON_IN:
        np.save(SAVE_DIR + f'eeg_raw_{run_tag}.npy',         eeg_buf)
        np.save(SAVE_DIR + f'aux_raw_{run_tag}.npy',         aux_buf)
    np.save(SAVE_DIR + f'trial_metadata_{run_tag}.npy', np.array(trial_metadata, dtype=object))
    np.save(SAVE_DIR + f'eeg_trials_{run_tag}.npy',     np.array(eeg_trials_list))
    np.save(SAVE_DIR + f'labels_{run_tag}.npy',          np.array(labels_list))
    print(f'[SAVE] Data saved to {SAVE_DIR}')

# ---------------------------------------------
#  PSYCHOPY WINDOW
# ---------------------------------------------
keyboard = kb.Keyboard()
win = visual.Window(
    size=[SCREEN_WIDTH, SCREEN_HEIGHT],
    fullscr=True,
    allowGUI=False,
    color=[-0.6, -0.6, -0.6],
    useRetina=False,
)

# ---------------------------------------------
#  STIMULI
# ---------------------------------------------
fixation = visual.TextStim(win, text='+', height=0.1, color='white')
stim_text = visual.TextStim(win, text='', height=0.15, color='white')

instruction_text = visual.TextStim(
    win, wrapWidth=1.5, height=0.07, color='white',
    text=(
        "FLANKER TASK\n\n"
        "Look at the CENTER arrow.\n\n"
        "<  ->  Press 'F' (Left)\n"
        ">  ->  Press 'J' (Right)\n\n"
        "Examples:\n"
        "<<<<<  (Press F)\n"
        ">><>>  (Press F)\n\n"
        "Press SPACE to begin."
    )
)

trial_counter = visual.TextStim(win, pos=(0, -0.9), height=0.05, color='grey', text='')
feedback_text = visual.TextStim(win, pos=(0, -0.5), height=0.07, text='')
photosensor = visual.Rect(win, width=0.1, height=0.1, pos=[0.9, -0.9], fillColor='black', lineWidth=0)

def show_photosensor(state):
    photosensor.fillColor = 'white' if state else 'black'

def draw_screen():
    photosensor.draw()
    trial_counter.draw()

# ---------------------------------------------
#  LOGIC & SEQUENCE
# ---------------------------------------------
def build_trial_sequence(n_con, n_inc, seed=0):
    rng = np.random.default_rng(seed)
    trials = []
    
    # Congruent
    for _ in range(n_con):
        is_left = rng.choice([True, False])
        if is_left: trials.append({'cond':'con', 'ans':'f', 'stim':'< < < < <'})
        else:       trials.append({'cond':'con', 'ans':'j', 'stim':'> > > > >'})
        
    # Incongruent
    for _ in range(n_inc):
        is_left = rng.choice([True, False])
        if is_left: trials.append({'cond':'inc', 'ans':'f', 'stim':'> > < > >'})
        else:       trials.append({'cond':'inc', 'ans':'j', 'stim':'< < > < <'})
        
    rng.shuffle(trials)
    for t in trials:
        t['iti'] = rng.uniform(ITI_MIN, ITI_MAX)
    return trials

def extract_response_locked_epoch(press_sample):
    if not CYTON_IN: return None
    pre  = int(EPOCH_PRE_PRESS  * SAMPLING_RATE)
    post = int(EPOCH_POST_PRESS * SAMPLING_RATE)
    start, end = press_sample - pre, press_sample + post
    
    if start < 0 or end > eeg_buf.shape[1]: return None
    
    raw = eeg_buf[:, start:end].copy()
    filt = mne.filter.filter_data(raw, SAMPLING_RATE, 1.0, 40.0, verbose=False)
    base_end = int(0.2 * SAMPLING_RATE)
    filt -= np.mean(filt[:, :base_end], axis=1, keepdims=True)
    return filt

# ---------------------------------------------
#  MAIN LOOP
# ---------------------------------------------
trial_metadata = []
eeg_trials = []
labels = []

instruction_text.draw()
win.flip()
event.waitKeys(keyList=['space', 'escape'])

sequence = build_trial_sequence(N_CONGRUENT, N_INCONGRUENT, seed=RUN)
n_trials = len(sequence)

print(f'[TASK] Run {RUN}: {n_trials} trials')

for i, trial in enumerate(sequence):
    trial_counter.text = f'{i+1}/{n_trials}'
    
    # 1. ITI (Inter-Trial Interval)
    fixation.draw()
    show_photosensor(False)
    draw_screen()
    win.flip()
    core.wait(trial['iti'])
    
    # 2. TRIAL EXECUTION (Non-blocking Loop)
    # --------------------------------------
    stim_onset = current_sample_index()
    trial_clock = core.Clock() # Reset clock 0.00s
    
    resp_key = None
    rt = None
    press_sample = None
    
    # Constant loop for the duration of RESPONSE_WINDOW
    while trial_clock.getTime() < RESPONSE_WINDOW:
        
        # A. Draw Logic (What should be on screen?)
        if trial_clock.getTime() < STIM_DURATION:
            stim_text.text = trial['stim']
            stim_text.draw()
            show_photosensor(True) # Sync marker ON
        else:
            stim_text.text = '' 
            show_photosensor(False) # Sync marker OFF
            
        draw_screen()
        win.flip() # Refresh screen (Approx every 16.7ms)
        
        # B. Key Logic (Did you press something?)
        keys = keyboard.getKeys(keyList=['f', 'j', 'escape'], waitRelease=False)
        
        for k in keys:
            if k.name == 'escape':
                if CYTON_IN:
                    stop_event.set()
                    board.stop_stream()
                    board.release_session()
                win.close()
                core.quit()
            
            # Record ONLY the first keypress
            if resp_key is None:
                resp_key = k.name
                rt = k.rt
                press_sample = stim_onset + int(rt * SAMPLING_RATE)
                # Optional: Break loop immediately after response?
                # No, better to keep timing consistent (fixed pace).
                
    # 3. Classify Outcome
    outcome = ''
    if resp_key is None:
        outcome = 'miss'
    elif resp_key == trial['ans']:
        outcome = 'correct'
    else:
        outcome = 'error' 
        
    # 4. Extract Epoch
    epoch = None
    if CYTON_IN and press_sample is not None:
        drain_queue()
        needed = press_sample + int(EPOCH_POST_PRESS * SAMPLING_RATE)
        timeout = 0
        while eeg_buf.shape[1] < needed and timeout < 20:
            time.sleep(0.01)
            drain_queue()
            timeout += 1
            
        epoch = extract_response_locked_epoch(press_sample)
        if epoch is not None:
            eeg_trials.append(epoch)
            labels.append(outcome)
            
    # Metadata
    trial_metadata.append({
        'idx': i,
        'cond': trial['cond'],
        'stim': trial['stim'],
        'ans': trial['ans'],
        'resp': resp_key,
        'outcome': outcome,
        'rt': rt
    })
    
    # Feedback
    feedback_text.text = outcome.upper()
    feedback_text.color = 'green' if outcome == 'correct' else 'red'
    feedback_text.draw()
    win.flip()
    core.wait(0.25)
    
    rt_disp = f"{rt:.3f}s" if rt else "---"
    print(f"Trial {i+1}: {trial['stim']} -> {outcome} ({rt_disp})")

# ---------------------------------------------
#  SAVE & EXIT
# ---------------------------------------------
save_all_data(trial_metadata, eeg_trials, labels)

summary = visual.TextStim(win, text="Run Complete.\nSaving...", height=0.1)
summary.draw()
win.flip()
core.wait(2.0)

if CYTON_IN:
    stop_event.set()
    board.stop_stream()
    board.release_session()

win.close()
core.quit()