# class for file dialog. To be used for selecting a case directory. actual case files loaded separately.
import tkinter as tk
from tkinter import filedialog as fd

class FileDialog():
    def __init__(self,initdir):
       self.initdir = initdir
       self.dir = self.initdir
       self.filenames = None

    def create_dialog(self):
        self.root = tk.Tk()
        self.root.title('File Dialog')
        self.root.resizable(False,False)
        self.root.geometry('300x150')

    def remove_dialog(self):
        self.root.destroy()
 
    def select_dir(self):
        self.create_dialog()
        self.dir = fd.askdirectory(
            title='Select data directory',
            initialdir=self.dir
        )
        self.remove_dialog()
        return

    def select_file(self):
        self.create_dialog()
        filetypes = (
            ('all files', '*.*'),
            ('nifti files', '*.nii*'),
            ('dicom files', '*.dcm')
        )

        self.filenames = fd.askopenfilenames(
            title='Select T1 and FLAIR images',
            initialdir=self.dir,
            filetypes=filetypes)
        return
       