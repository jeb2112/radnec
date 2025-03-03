import sys
from fastapi import FastAPI

# from twisted.internet import tksupport, reactor, protocol
# import tk_async_execute as tae
from tkinter import *

app = FastAPI()

class BackEnd():

    def __init__(self, optionsFlag = 0, debug=False):
        try:

            self.root = Tk()

            # start a loop for asyncio. might not need this here though.
            # tae.start()
            # tiwsted isn't compatible with the tae loop so just use mainloop()
            if False:
                tksupport.install(self.root)
                reactor.run()
            else:
                self.root.mainloop()
        except Exception as e:
            self.config.logger.error("{}: {}".format(e.args[0], sys.exc_info()[0]))
            print("{}: {}".format(e.args[0], sys.exc_info()[0]))
        else:
            print("Exit")

    def windowCloseHandler(self):
        if False:
            reactor.stop()
        # tae.stop()
        self.root.quit()
        self.root.destroy()

class Test():
    def __init__(self):
        self.test = 1
        with open('/home/src/no-docker/test.txt','w') as fp:
            fp.write('test = {}'.format(self.test))


@app.get("/")
async def root():
    # backend = BackEnd()
    test = Test()
    return {"message": "test complete"}
