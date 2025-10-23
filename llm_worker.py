from PyQt6.QtCore import (
    QThread, pyqtSignal
)
import requests
import sseclient
import json

class LLMWorker(QThread):
    result_ready = pyqtSignal(str)
    token_emit = pyqtSignal(str)
    error_emit = pyqtSignal(str)
    stats_emit = pyqtSignal(dict)

    def __init__(self, request, url):
        super().__init__()
        self.request = request
        self.reply = ""
        self._is_running = True
        self.url = url
        print("LLMWorker is created")

    def run(self):
        print("LLMWorker is running")
        self._is_running = True
        try:
            response = requests.post(self.url, json=self.request, stream=True)
            client = sseclient.SSEClient(response)
            self.reply = ""
            for event in client.events():
                print(f'Event: {event.data}')
                if event.data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(event.data)
                    print(f'Chunk: {chunk}')
                    choices = chunk.get('choices', [])
                    print(f'Choices: {choices}')
                    if not choices:
                        if "usage" in chunk or "timings" in chunk:
                            self.stats_emit.emit(chunk)
                        continue
                    delta = choices[0].get('delta', {})
                    token = delta.get('content')
                    if token:
                        self.reply += token
                        self.token_emit.emit(token)
                except (json.JSONDecodeError, KeyError):
                    continue
            self.result_ready.emit(self.reply)

            if not self._is_running:
                return
        except Exception as e:
            self.error_emit.emit(str(e))
        
    def stop(self):
        self._is_running = False