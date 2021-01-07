# ******************** events for All Buttons and Frames and shows on the status bar ***********************************
def train_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Click to train CNN model'


def start_app(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Click to start application'


def train_data_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Select train data'


def valid_data_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Select validation data'


def test_data_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Select test data'


def train_btn_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Reset  data and train'


def clear_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Clears selected file(s)'


def back_welcome_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Go back to main menu'


def test_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Run the system and test its accuracy'


def file_test_text_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Select text file for testing'


def file_test_values_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Select the value file for testing'


def test_btn_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Run test on selected files'


def eval_text_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = "Evaluate input file that hasn't been pre-labelled"


def gui_eval_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Click to get emotion from text'


def predict_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Predicts the emotion from the inputted text'


def clear_predict_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Clears the text'


def file_eval_text_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Select a text file to evaluate'


def eval_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Evaluates a file that hasn\'t been pre-labelled'


def info_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'For more information'


def exit_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Close application'


def self_event(event, s_bar):
    """
    :param event: event handler for Tkinter Python - binds the mouse to display a text
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Click a button to continue'


def print_event(s_bar):
    """
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Now Printing................................'


def save_event(s_bar):
    """
    :param s_bar: reference to status_bar
    :return: nothing
    """
    s_bar['text'] = 'Saving files..................................'

# **************************************************** End of mouse movements ******************************************
