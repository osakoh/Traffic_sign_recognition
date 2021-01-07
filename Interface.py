import tkinter as tk
import os
import numpy as np

from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
from tkinter.messagebox import showinfo, askokcancel

# initialise text file and value file to empty
train_file = ""
valid_file = ""
test_file = ""


# ************************************************* MainWindow *********************************************************
class MainWindow(tk.Tk):
    """
    This is the Main frame. It hold the other frames
    """

    def __init__(self, *args, **kwargs):
        # Call to __init__ of super class
        tk.Tk.__init__(self, *args, **kwargs)

        global container

        # change the default tk icon
        tk.Tk.iconbitmap(self, default="img/logo.ico")

        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand="True")

        container.grid_rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # empty list - key & values
        self.frames = {}

        for F in (WelcomeFrame, StartFrame, TrainFrame, TestFrame, GUIFrame, EvaluateFrame):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            # **************** Move movement showing on status bar for frame ****
            frame.bind("<Motion>", self_event)
            # *******************************************************************

        # show the start(Main Menu) frame
        self.show_cur_frame(WelcomeFrame)

    def show_cur_frame(self, current):
        """
        :param current: this function takes current as the dictionary key
        :return: a single frame
        """
        frame = self.frames[current]
        frame.tkraise()


# ************************************************* End of MainWindow **************************************************


# ******************************************** Welcome frame ****************************************************
class WelcomeFrame(tk.Frame):
    """
    The first frame shown when the application is ran.
    """

    def __init__(self, parent, controller):

        # 'parent class' is 'MainWindow'
        tk.Frame.__init__(self, parent, background='gray')
        global status_bar

        # ************************* Traffic label ********************************
        tsr_label = tk.Label(self, text="Traffic Sign Recognition", width=20)
        tsr_label.place(x=5, y=2)
        tsr_label.config(bd=2, font=("Arial BOLD", 14), bg='gray')

        name_label = tk.Label(self, text="By: 18022563", width=10)
        name_label.place(x=20, y=28)
        name_label.config(bd=2, font=("Arial BOLD", 13), bg='gray')
        # ************************* End of Traffic label ********************************

        # ************************* Traffic sign image ********************************
        load = Image.open("img/ts_logo.png")
        render = ImageTk.PhotoImage(load)
        img_label = tk.Label(self, width=300, height=215, image=render)
        img_label.image = render
        img_label.place(x=5, y=55)
        # ************************* End of Traffic sign image ********************************

        # ************************** button to the left ********************************
        # training button
        continue_btn = tk.Button(self, text="Continue", width=13, command=lambda: controller.show_cur_frame(StartFrame))
        continue_btn.grid(row=0, column=1, padx=350, pady=50, sticky="e")
        continue_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        continue_btn.bind("<Motion>", start_app)

        # ****************** Exit application through exit button *****************

        def exit_close_btn():
            answer = askokcancel("Exit application!", "Are you sure?")

            if answer is True:
                self.quit()
            else:
                return False

        # Exit button
        exit_btn = tk.Button(self, text="Exit", width=13, command=exit_close_btn)
        exit_btn.grid(row=2, column=1, padx=10, pady=0)
        exit_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), fg='red')
        exit_btn.bind("<Motion>", exit_event)


# ***************************************** End of Welcome frame *******************************************************

# ******************************************** StartPage(Main Menu) ****************************************************
class StartFrame(tk.Frame):

    def __init__(self, parent, controller):

        # 'parent class' is 'WelcomeWindow'
        tk.Frame.__init__(self, parent, background='gray')
        global status_bar

        # ************************** button to the left ********************************
        # training button
        train_btn = tk.Button(self, text="Training", width=13, command=lambda: controller.show_cur_frame(TrainFrame))
        train_btn.grid(row=0, column=0, padx=68, pady=25)
        train_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        train_btn.bind("<Motion>", train_event)

        # testing button
        test_btn = tk.Button(self, text="Testing", width=13, command=lambda: controller.show_cur_frame(TestFrame))
        test_btn.grid(row=1, column=0, pady=24)
        test_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        test_btn.bind("<Motion>", test_event)

        # evaluate text button
        eval_text_btn = tk.Button(self, text="Evaluate Text", width=13,
                                  command=lambda: controller.show_cur_frame(EvaluateFrame))
        eval_text_btn.grid(row=2, column=0, pady=25)
        eval_text_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        eval_text_btn.bind("<Motion>", eval_text_event)

        # ************************** button to the right ********************************

        # gui evaluation button
        gui_eval_btn = tk.Button(self, text="Predict Sign", width=13,
                                 command=lambda: controller.show_cur_frame(GUIFrame))
        gui_eval_btn.grid(row=0, column=1, padx=40)
        gui_eval_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        gui_eval_btn.bind("<Motion>", gui_eval_event)

        # more information evaluation button and function
        def more_info():
            text = ''' '''
            showinfo('More Information', text)

        more_info_btn = tk.Button(self, text="Information", width=13, command=more_info)
        more_info_btn.grid(row=1, column=1, padx=40)
        more_info_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        more_info_btn.bind("<Motion>", info_event)

        # ****************** Close window through exit button *****************

        def exit_btnclose():
            answer = askokcancel("Exit application!", "Are you sure?")

            if answer is True:
                self.quit()
                return
            else:
                return False

        # Exit button
        # exit_btn = tk.Button(self, text="Exit", width=13, command=exit_btnclose)
        exit_btn = tk.Button(self, text="Back to start", width=13,
                             command=lambda: controller.show_cur_frame(WelcomeFrame))
        exit_btn.grid(row=2, column=1, padx=40, pady=24)
        exit_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), fg='brown')
        exit_btn.bind("<Motion>", exit_event)


# ***************************************** End of Start Page **********************************************************

# # ******************** Events for All Buttons and Frames and shows on the status bar *********************************
def train_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Click to train CNN model'


def start_app(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Click to start application'


def train_data_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Select train data'


def valid_data_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Select validation data'


def test_data_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Select test data'


def train_btn_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Reset  data and train'


def clear_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Clears selected file(s)'


def back_welcome_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Go back to main menu'


def test_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Run the system and test its accuracy'


def file_test_text_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Select text file for testing'


def file_test_values_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Select the value file for testing'


def test_btn_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Run test on selected files'


def eval_text_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = "Evaluate input file that hasn't been pre-labelled"


def gui_eval_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Click to get emotion from text'


def predict_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Predicts the emotion from the inputted text'


def clear_predict_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Clears the text'


def file_eval_text_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Select a text file to evaluate'


def eval_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Evaluates a file that hasn\'t been pre-labelled'


def info_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'For more information'


def exit_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Close application'


def self_event(event):
    """
    displays text on the status bar
    """
    status_bar['text'] = 'Click a button to continue'


# # **************************************************** End of mouse movements ****************************************

# ******************************************************* load data for training ***************************************
def load_train_data():
    global train_file
    current_path = "./data/"
    train_file = filedialog.askopenfilename(title='Choose text file',
                                            initialdir=current_path,
                                            filetypes=[("Pickle files", "*.p")])
    train_data_label.config(text=os.path.basename(train_file))


# **************************************************** load data for validation **********************************
def load_valid_data():
    global valid_file
    current_path = "./data/"
    valid_file = filedialog.askopenfilename(title='Choose value file',
                                            initialdir=current_path,
                                            filetypes=[("Pickle files", "*.p")])

    valid_data_label.config(text=os.path.basename(valid_file))


# **************************************************** load data for testing **********************************
def load_test_data():
    global test_file
    current_path = "./data/"
    test_file = filedialog.askopenfilename(title='Choose value file',
                                           initialdir=current_path,
                                           filetypes=[("Pickle files", "*.p")])

    test_data_label.config(text=os.path.basename(test_file))


# ************************************* End of load tweet text and value for training **********************************

# **************************************************** Read files for training *****************************************
def train_model():
    pass


# ********************************************* End of Read files for training *****************************************


# *************************************************** progressNotice Function ******************************************
def progressNotice():
    showinfo('Info', "Process completed!")


# ************************************************ End of progressNotice ***********************************************


# ***************************************** Show training frame ********************************************************
class TrainFrame(tk.Frame):

    def __init__(self, parent, controller):
        # 'parent class' is 'WelcomeWindow'
        tk.Frame.__init__(self, parent, background='gray')
        global train_data_label, valid_data_label, test_data_label

        # ************************** train button and entry *******************
        train_data_btn = tk.Button(self, text="Train data", width=12, command=load_train_data)
        # train_data_btn.grid(row=0, column=0, padx=5, pady=30)
        train_data_btn.place(x=10, y=10)

        train_data_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        train_data_btn.bind("<Motion>", train_data_event)

        train_data_label = tk.Label(self, width=25)
        # train_data_label.grid(row=0, column=1, ipady=5, pady=20)
        train_data_label.place(x=160, y=16)
        train_data_label.config(bd=2, font=("Arial ITALIC", 13))
        # ********************End of train button and entry *******************

        # ******************** validation button and entry *******************
        valid_data_btn = tk.Button(self, text="Validation data", width=12, command=load_valid_data)
        # valid_data_btn.grid(row=3, column=0, pady=28)
        valid_data_btn.place(x=10, y=70)
        valid_data_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        valid_data_btn.bind("<Motion>", valid_data_event)
        #
        valid_data_label = tk.Label(self, width=25)
        # valid_data_label.grid(row=3, column=1, ipady=5)
        valid_data_label.place(x=160, y=75)
        valid_data_label.config(bd=2, font=("Arial ITALIC", 13))
        # ********************************End of validation button and entry *************************

        # ******************** test button and entry *******************
        test_data_btn = tk.Button(self, text="Test data", width=12, command=load_test_data)
        # test_data_btn.grid(row=3, column=0, pady=28)
        test_data_btn.place(x=10, y=134)
        test_data_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        test_data_btn.bind("<Motion>", test_data_event)

        test_data_label = tk.Label(self, width=25)
        # test_data_label.grid(row=3, column=1, ipady=5)
        test_data_label.place(x=160, y=140)
        test_data_label.config(bd=2, font=("Arial ITALIC", 13))
        # ********************************End of test button and entry *************************

        # 3 buttons: Train, Clear, and Back button
        # training button
        train_btn = tk.Button(self, text="Train", width=8, command=train_model)
        # train_btn.grid(row=4, column=0)
        train_btn.place(x=25, y=200)
        train_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), fg='red')
        train_btn.bind("<Motion>", train_event)

        # testing button
        back_btn = tk.Button(self, text="Back", width=8, command=lambda: controller.show_cur_frame(StartFrame))
        # back_btn.grid(row=4, column=2)
        back_btn.place(x=175, y=200)
        back_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        back_btn.bind("<Motion>", back_welcome_event)

        # clear button for training frame
        def clear_training():
            train_data_label['text'] = ""
            valid_data_label['text'] = ""
            test_data_label['text'] = ""

        clear_train_btn = tk.Button(self, text="Clear", width=8, command=clear_training)
        # clear_train_btn.grid(row=4, column=1)
        clear_train_btn.place(x=325, y=200)
        clear_train_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        clear_train_btn.bind("<Motion>", clear_event)


# ******************************************** End of Training Frame ***************************************************


# ********************************************* Show test frame ********************************************************

#  load tweet text for test ***************************************
def testTweetText():
    global train_file
    current_path = "./data/"
    train_file = filedialog.askopenfilename(title='Choose text file', initialdir=current_path, filetypes=[("CSV files",
                                                                                                           "*.csv")])

    test_tweet_txt_abel.config(text=os.path.basename(train_file))


# end of load tweet text for test **********************************


#  load tweet values for test **************************************
def testTweetValues():
    global valid_file
    current_path = "./data/"
    valid_file = filedialog.askopenfilename(title='Choose value file', initialdir=current_path, filetypes=[("CSV files"
                                                                                                            ,
                                                                                                            "*.csv")])

    test_tweet_values_label.config(text=os.path.basename(valid_file))


# end of load tweet values for test ***********************************


#  Read files for testing ************************************************
def test():
    pass


# end of Read files for test ********************************************


class TestFrame(tk.Frame):

    def __init__(self, parent, controller):
        # 'parent class' is 'WelcomeWindow'
        tk.Frame.__init__(self, parent, background='gray')
        global test_tweet_txt_abel, test_tweet_values_label

        # ************************** tweet text button and entry *******************
        test_tweet_txt_btn = tk.Button(self, text="Tweet text", width=12, command=testTweetText)
        test_tweet_txt_btn.grid(row=0, column=0, padx=5, pady=35)
        test_tweet_txt_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        test_tweet_txt_btn.bind("<Motion>", file_test_text_event)

        test_tweet_txt_abel = tk.Label(self, width=25)
        test_tweet_txt_abel.grid(row=0, column=1, ipady=5, pady=20)
        test_tweet_txt_abel.config(bd=2, font=("Arial ITALIC", 13))
        # ********************End of tweet Text button and label *******************

        # *********************** tweet values button and label ********************
        test_tweet_values_btn = tk.Button(self, text="Tweet values", width=12, command=testTweetValues)
        test_tweet_values_btn.grid(row=3, column=0, pady=30)
        test_tweet_values_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        test_tweet_values_btn.bind("<Motion>", file_test_values_event)

        test_tweet_values_label = tk.Label(self, width=25)
        test_tweet_values_label.grid(row=3, column=1, ipady=5)
        test_tweet_values_label.config(bd=2, font=("Arial ITALIC", 13))
        # ********************End of tweet values button and label *******************

        # test button
        test_btn = tk.Button(self, text="Test", width=12, command=test)
        test_btn.grid(row=4, column=0)
        test_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), fg='red')
        test_btn.bind("<Motion>", test_btn_event)

        # back button
        back_btn = tk.Button(self, text="Back", width=10, command=lambda: controller.show_cur_frame(StartFrame))
        back_btn.grid(row=4, column=2)
        back_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        back_btn.bind("<Motion>", back_welcome_event)

        # clear button for test frame
        def clear_test():
            test_tweet_txt_abel['text'] = ""
            test_tweet_values_label['text'] = ""

        clear_test_btn = tk.Button(self, text="Clear", width=8, command=clear_test)
        clear_test_btn.grid(row=4, column=1)
        clear_test_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        clear_test_btn.bind("<Motion>", clear_event)


# *********************************************** End of test Frame ****************************************************


# ********************************************* Show evaluate frame ****************************************************

#  load tweet text for test ***************************************
def evaluate_tweet_text():
    global train_file
    current_path = "./data/"
    train_file = filedialog.askopenfilename(title='Choose text file for evaluation', initialdir=current_path,
                                            filetypes=[("CSV files", "*.csv")])

    eval_tweet_txt_label.config(text=os.path.basename(train_file))


# end of load tweet text for test **********************************


#  Read files for text evaluation ************************************************
def evaluate():
    pass


# end of Read files for evaluate ********************************************


class EvaluateFrame(tk.Frame):

    def __init__(self, parent, controller):
        # 'parent class' is 'WelcomeWindow'
        tk.Frame.__init__(self, parent, background='gray')
        global eval_tweet_txt_label
        # ************************** tweet text button and entry *******************
        eval_tweet_txt_btn = tk.Button(self, text="Text Evaluate", width=12, command=evaluate_tweet_text)
        eval_tweet_txt_btn.grid(row=0, column=0, padx=5, pady=35)
        eval_tweet_txt_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 12), activeforeground='gray')
        eval_tweet_txt_btn.bind("<Motion>", file_eval_text_event)

        eval_tweet_txt_label = tk.Label(self, width=25)
        eval_tweet_txt_label.grid(row=0, column=1, ipady=5, pady=20)
        eval_tweet_txt_label.config(bd=2, font=("Arial ITALIC", 13))
        # ********************End of tweet Text button and label *********************

        # test button
        evaluate_btn = tk.Button(self, text="Evaluate", width=12, command=evaluate)
        evaluate_btn.grid(row=4, column=0)
        evaluate_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), fg='red')
        evaluate_btn.bind("<Motion>", eval_event)

        # back button
        back_btn = tk.Button(self, text="Back", width=10, command=lambda: controller.show_cur_frame(StartFrame))
        back_btn.grid(row=4, column=2)
        back_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        back_btn.bind("<Motion>", back_welcome_event)

        # clear button for test frame
        def clear_eval_text():
            eval_tweet_txt_label['text'] = ""

        clear_eval_btn = tk.Button(self, text="Clear", width=8, command=clear_eval_text)
        clear_eval_btn.grid(row=4, column=1)
        clear_eval_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        clear_eval_btn.bind("<Motion>", clear_event)


# ********************************************* End of text evaluation Frame *******************************************


# ******************************************* show GUI Evaluate frame **************************************************
class GUIFrame(tk.Frame):
    def __init__(self, parent, controller):
        # 'parent class' is 'WelcomeWindow'
        tk.Frame.__init__(self, parent, background='gray')
        input_label = tk.Label(self, text="Input text", bg='gray')
        input_label.grid(row=0, column=0, padx=3, pady=20, sticky="W")
        input_label.config(font=("Arial", 14))

        input_str = tk.Entry(self, width=46)
        input_str.grid(row=0, column=1, ipady=6, pady=10)
        input_str.config(font=("Arial", 13))

        predicted_label = tk.Label(self, text="Predicted:", bg='gray')
        predicted_label.grid(row=1, column=0)
        predicted_label.config(font=("Arial", 14))

        v = tk.StringVar()
        output = tk.Label(self, textvariable=v, font=("Arial", 15), bg='gray')
        output.grid(row=1, column=1, sticky="W", pady=20)
        output.config(font=("Arial", 14))

        # predict function for predict button ************************************************
        def pred_button():
            pass

        # predict button *************************************************
        predict_btn = tk.Button(self, text="Predict", command=pred_button)
        predict_btn.grid(row=2, column=1, sticky="nsew", pady=10)
        predict_btn.config(relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='red')
        predict_btn.bind("<Motion>", predict_event)

        # back button for GUI frame
        back_btn = tk.Button(self, text="Back to main menu", command=lambda: controller.show_cur_frame(StartFrame))
        back_btn.grid(row=3, column=1, sticky="nwes")
        back_btn.config(bd=2, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        back_btn.bind("<Motion>", back_welcome_event)

        # clear button for GUI frame
        def clear_text_entry():
            v.set("")
            input_str.delete(0, 'end')

        # clear btn for GUI frame
        clear_test_btn = tk.Button(self, text="Clear", command=clear_text_entry)
        clear_test_btn.grid(row=2, column=0, padx=10)
        clear_test_btn.config(bd=3, relief=tk.RAISED, font=("Arial Bold", 13), activeforeground='gray')
        clear_test_btn.bind("<Motion>", clear_predict_event)


# ******************************************* End of GUI Evaluate frame ************************************************


# ********************************************* running the main class *************************************************
root = MainWindow()

# ******************************************************* Menu Bar *****************************************************
# Insert a menu bar on the main window
menubar = tk.Menu(root)

# ********************************* Creates a menu button labeled "File" and "Quit" ************************************
file_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=file_menu)
menubar.add_cascade(label='Quit', command=root.quit)


# *************************************** Event to show on status bar **************************************************
def print_event(): status_bar['text'] = 'Now Printing................................'


def save_event(): status_bar['text'] = 'Saving files..................................'


# ********************************************* Creates  "File" Sub-menus **********************************************
printStatus = file_menu.add_command(label='Print', command=print_event)
saveStatus = file_menu.add_command(label='Save', command=save_event)

# ************************************************ End of menu *********************************************************

# ******************************************************** status bar **************************************************
status_bar_frame = tk.Frame(container, bd=1, relief=tk.SUNKEN)
status_bar_frame.grid(row=4, column=0, columnspan=6, sticky="we")

status_bar = tk.Label(status_bar_frame, text="Welcome", bg="#dfdfdf", anchor=tk.W)

status_bar.pack(side=tk.BOTTOM, fill=tk.X)
# status_bar.config(anchor=tk.W, font=("Times", 11))
status_bar.config(font=("Times", 11))
status_bar.grid_propagate(0)
# ********************************************* End of status bar ******************************************************

# ************************************************** Centralise the window *********************************************
window_height = 300
window_width = 520
# specifies width and height of window1
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# specifies the co-ordinates for the screen
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
root.title("TSR | Thesis")  # Add a title
root.resizable(False, False)  # This code helps to disable windows from resizing
root.config(menu=menubar)
root.mainloop()
