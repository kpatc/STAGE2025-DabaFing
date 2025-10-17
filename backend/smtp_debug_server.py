# smtp_debug_server.py
import smtpd
import asyncore
import threading

class SMTPDebugServer(smtpd.DebuggingServer, threading.Thread):
    def __init__(self, localaddr):
        threading.Thread.__init__(self)
        self.localaddr = localaddr
        self.daemon = True

    def run(self):
        smtpd.DebuggingServer(self.localaddr, None)
        try:
            asyncore.loop()
        except Exception as e:
            print(f"Erreur SMTP server: {e}")

def start_smtp_debug_server():
    server = SMTPDebugServer(('localhost', 8025))
    server.start()
