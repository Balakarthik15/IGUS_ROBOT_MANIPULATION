#!/usr/bin/env python3
"""
Small window with OPEN and CLOSE buttons for the gripper.

    python3 gripper_ui.py

CLOSE: output 31 off, output 30 on, after 5 s output 30 off again.
OPEN : output 30 off, output 31 on, after 5 s output 31 off again.
Keys: c = close, o = open.

Same as typing:
  ros2 service call /set_digital_output igus_rebel_msgs/srv/SetDigitalOutput "{output: {output: 30, is_on: true}}"
"""
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from igus_rebel_msgs.srv import SetDigitalOutput

SERVICE = "/set_digital_output"
OUT_CLOSE, OUT_OPEN = 30, 31
PULSE_MS = 5000


class Io(Node):
    def __init__(self):
        super().__init__("gripper_ui")
        self.cli = self.create_client(SetDigitalOutput, SERVICE)

    def set(self, output, on, done=None):
        if not self.cli.service_is_ready():
            if done:
                done(False, "service %s not available" % SERVICE)
            return
        req = SetDigitalOutput.Request()
        req.output.output = output
        req.output.is_on = on
        fut = self.cli.call_async(req)

        def cb(f):
            try:
                r = f.result()
                ok, msg = r.success, r.message
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, str(exc)
            if done:
                done(ok, msg)
        fut.add_done_callback(cb)


class App:
    def __init__(self, root, io):
        self.root, self.io = root, io
        self.timer = None
        root.title("Gripper")
        root.resizable(False, False)
        f = tk.Frame(root, padx=16, pady=16)
        f.pack()
        self.b_open = tk.Button(f, text="OPEN", width=10, height=2, font=("Sans", 14, "bold"),
                                command=lambda: self.pulse(OUT_OPEN, OUT_CLOSE, "opening"))
        self.b_close = tk.Button(f, text="CLOSE", width=10, height=2, font=("Sans", 14, "bold"),
                                 command=lambda: self.pulse(OUT_CLOSE, OUT_OPEN, "closing"))
        self.b_open.grid(row=0, column=0, padx=6)
        self.b_close.grid(row=0, column=1, padx=6)
        self.status = tk.Label(f, text="ready", width=34, anchor="w", font=("Sans", 10))
        self.status.grid(row=1, column=0, columnspan=2, pady=(12, 0))
        root.bind("o", lambda e: self.b_open.invoke())
        root.bind("c", lambda e: self.b_close.invoke())
        root.protocol("WM_DELETE_WINDOW", self.quit)

    def show(self, text):
        self.root.after(0, lambda: self.status.config(text=text))

    def pulse(self, on_out, other_out, word):
        if self.timer is not None:           # a new click cancels the running pulse
            self.root.after_cancel(self.timer)
            self.timer = None
        self.show("%s ..." % word)
        # first switch the opposite output off, then this one on
        self.io.set(other_out, False)
        self.io.set(on_out, True, lambda ok, m: self.show(
            "%s: output %d ON, off in %d s" % (word, on_out, PULSE_MS // 1000) if ok else "ERROR: %s" % m))
        self.timer = self.root.after(PULSE_MS, lambda: self.off(on_out, word))

    def off(self, out, word):
        self.timer = None
        self.io.set(out, False, lambda ok, m: self.show(
            "%s done, output %d OFF" % (word, out) if ok else "ERROR switching off: %s" % m))

    def quit(self):
        # leave both outputs off when the window is closed
        self.io.set(OUT_CLOSE, False)
        self.io.set(OUT_OPEN, False)
        self.root.after(300, self.root.destroy)


def main():
    rclpy.init()
    io = Io()
    threading.Thread(target=rclpy.spin, args=(io,), daemon=True).start()
    root = tk.Tk()
    App(root, io)
    if not io.cli.wait_for_service(timeout_sec=2.0):
        print("warning: %s not found yet. Is the robot driver running?" % SERVICE)
    root.mainloop()
    io.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
