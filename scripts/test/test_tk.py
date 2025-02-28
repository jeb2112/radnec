from tkinter import ttk
import tkinter as tk

style = ttk.Style()
style.configure('debug.TFrame',background='green')
style.configure('debug2.TFrame',background='red')

def button_clicked():
    print('clicked')

class subFrame():
    def __init__(self,frame):
        self.frame = ttk.Frame(frame,style='debug.TFrame',padding='0')
        self.frame.grid()

class subsubFrame(subFrame):
    def __init__(self,frame):
        super().__init__(frame)

        self.myframe = ttk.Frame(self.frame,style='debug2.TFrame',padding='0')
        bnt = ttk.Button(self.myframe, text="Click 2",command=button_clicked, width=10)
        bnt.grid(row=0,column=0,sticky='w')


if __name__ == "__main__":
    root = tk.Tk()
    main_frame = ttk.Frame(root,style='debug.TFrame')
    main_frame.grid(row=0,column=0)
    bnt = ttk.Button(main_frame, text="Click me", command=button_clicked, width=20)
    bnt.grid(row=0,column=0)

    subsubframe = subsubFrame(main_frame)
    subsubframe.myframe.grid(row=1,column=0,sticky='w')
    root.mainloop()  # Main Tkinter loop
    print('DONE')