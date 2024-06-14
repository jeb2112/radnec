import asyncio
import concurrent.futures

import sounddevice
import os
import boto3
import signal
import functools
import concurrent

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self):
        self.transcript = []
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                self.transcript.append(alt.transcript)
                print(alt.transcript)


class Transcription(object):
    def __init__(self,root):

        credentials = boto3.Session(profile_name='jbishop-dev').get_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token
        self.transcript = None
        self.tasks = None
        self.root = root # tkinter root in case its needed

    async def mic_stream(self):
        # This function wraps the raw input stream from the microphone forwarding
        # the blocks to an asyncio.Queue.
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        # Be sure to use the correct parameters for the audio stream that matches
        # the audio formats described for the source language you'll be using:
        # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
        stream = sounddevice.RawInputStream(
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=1024 * 2,
            dtype="int16",
        )
        # Initiate the audio stream and asynchronously yield the audio chunks
        # as they become available.
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status


    async def write_chunks(self,stream):
        # This connects the raw audio chunks generator coming from the microphone
        # and passes them along to the transcription stream.
        try:
            async for chunk, status in self.mic_stream():
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
                # await asyncio.to_thread(stream.input_stream.send_audio_event,audio_chunk=chunk)
            await stream.input_stream.end_stream()
        except asyncio.CancelledError as e:
            print('cancelled')
            raise

    def ask_exit(self,signame='SIGINT'):
        print('got signal {}: exit'.format(signame))
        self.loop.stop()

    async def basic_transcribe(self):
        # Setup up our client with our chosen AWS region
        client = TranscribeStreamingClient(region="us-west-2")
        self.loop = asyncio.get_running_loop()

        # for CtrlC, add this keyboard handler.
        # however, added handlers like this aren't compatiable with the tkinter-async-execute module
        # get a set_wakeup_fd error. 
        if False:
            for signame in {'SIGINT','SIGTERM'}:
                loop.add_signal_handler(
                    getattr(signal,signame),
                    functools.partial(self.ask_exit,signame,loop)
                )

        # Start transcription to generate our async stream
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        # Instantiate our handler and start processing events
        # with the tkinter-async-execute module though no errors
        # need to be handled anymore
        handler = MyEventHandler(stream.output_stream)
        self.tasks =  asyncio.gather(self.write_chunks(stream), handler.handle_events())
        try:
            await self.tasks
        # Cancelled errors have to be handled when using CtrlC to exit the asyncio event loop
        except asyncio.CancelledError as e:
            print('cancelled')
            raise
        # runtime errors may also be raised.
        except RuntimeError as e:
            print('other')
            raise
        # a keyboard interrupt actually isn't raised from CtrlC in the async event loop
        except KeyboardInterrupt as e:
            print('ctrlC')
            raise
