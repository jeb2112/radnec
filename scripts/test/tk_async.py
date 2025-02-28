from tkinter import ttk
import tkinter as tk
import asyncio

import tk_async_execute as tae


async def async_function():
    # Call tkinter widget methods.
    print("Disabling button")
    tae.tk_execute(bnt.config, state="disabled")  # Thread safe exection
    await asyncio.sleep(5)
    print("Enabling button")
    tae.tk_execute(bnt.config, state="normal")
    await asyncio.sleep(2)

    # Change tkinter text
    print("Renaming button")
    tae.tk_execute(bnt.config, text="Example 2")
    await asyncio.sleep(2)


def button_clicked():
    # Call async function
    tae.async_execute(async_function(), wait=True, visible=True, pop_up=True, callback=None, master=root)

    # Close application
    root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    bnt = ttk.Button(root, text="Click me", command=button_clicked, width=20)
    bnt.pack()

    tae.start()  # Starts the asyncio event loop in a different thread.
    root.mainloop()  # Main Tkinter loop
    tae.stop()  # Stops the event loop and closes it.
    print('DONE')