from rknn.api import RKNN
from collections import deque
import sys
import time
import svgwrite
import threading
import numpy as np
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import GLib, GObject, Gst, GstBase

GObject.threads_init()
Gst.init(None)

class Inference:
    def __init__(self, gst_pipeline, inference_cb, src_size):
        self.inference_cb = inference_cb
        self.running = False
        self.gstbuffer = None
        self.sink_size = None
        self.src_size = src_size
        self.condition = threading.Condition()

        self.gst_pipeline = Gst.parse_launch(gst_pipeline)
        self.overlay = self.gst_pipeline.get_by_name('overlay')
        appsink = self.gst_pipeline.get_by_name('appsink')
        appsink.connect('new-sample', self.on_new_sample)

        bus = self.gst_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

    def run(self):
        try:
            # Start inference worker.
            self.running = True
            worker = threading.Thread(target=self.inference_loop)
            worker.start()

            # Run gst pipeline to capture frames
            self.gst_pipeline.set_state(Gst.State.PLAYING)

            # blocking on worker
            worker.join()

        except (KeyboardInterrupt, SystemExit) as e:
            # Clean up resources
            self.gst_pipeline.set_state(Gst.State.NULL)
            with self.condition:
                self.running = False
                self.condition.notify_all()
            print("Clean Up")
            raise e



    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            pass
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('Warning: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('Error: %s: %s\n' % (err, debug))
        return True

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if not self.sink_size:
            s = sample.get_caps().get_structure(0)
            self.sink_size = (s.get_value('width'), s.get_value('height'))
        with self.condition:
            self.gstbuffer = sample.get_buffer()
            self.condition.notify_all()
        return Gst.FlowReturn.OK


    def inference_loop(self):
        while True:
            with self.condition:
                while not self.gstbuffer and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                gstbuffer = self.gstbuffer
                self.gstbuffer = None

            ret, mapinfo = gstbuffer.map(Gst.MapFlags.READ)

            if ret:
                input_tensor = np.reshape(np.frombuffer(mapinfo.data, dtype=np.uint8), (224, 224, 3))
                gstbuffer.unmap(mapinfo)

                svg = self.inference_cb(input_tensor, self.src_size)
                if self.overlay:
                    self.overlay.set_property('data', svg)

if __name__ == "__main__":
    rknn = RKNN()

    print("Loading model")
    rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2')
    rknn.load_rknn('./mobilenet_v1.rknn')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        sys.exit(ret)


    # Gstreamer pipeline with two branches, one for inference and another for streaming over UDP with overlaid score/label
    # Change udpsink host and port for your setup
    pipeline_template = """
     rkv4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width={src_width},height={src_height},framerate={frame_rate}/1 
     ! tee name=t
        t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! videoscale 
           ! video/x-raw,width={scaled_width},height={scaled_height}
           ! videobox name=box autocrop=true
           ! video/x-raw,format=RGB,width={model_input_width},height={model_input_height} 
           ! appsink name=appsink emit-signals=true max-buffers=1 drop=true
        t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert
           ! rsvgoverlay name=overlay ! videoconvert 
           ! video/x-raw,format=NV12,width={src_width},height={src_height} 
           ! jpegenc ! rtpjpegpay ! queue ! udpsink  host=192.168.3.5 port=1234 sync=false
    """

    src_size         = (320, 240) # Camera source
    frame_rate       = 60
    model_input_size = (224, 224) # Model was trained on 224,224 
    # Scale ratio for padding or crop
    ratio            = min(model_input_size[0]/src_size[0], model_input_size[1]/src_size[1])
    scaled_size      = (int(src_size[0] * ratio), int(src_size[1] * ratio)) 

    gst_pipeline = pipeline_template.format(
        src_width=src_size[0], 
        src_height=src_size[1], 
        frame_rate=frame_rate,
        model_input_width=model_input_size[0], 
        model_input_height=model_input_size[1],
        scaled_width=scaled_size[0],
        scaled_height=scaled_size[1],
    )

    labels = [''] # keep index 0 blank
    with open('labels.txt', 'r') as f:
        for line in f:
            labels.append(line.strip().split(', ')[0])

    def average_fps(size):
        # Keep last frames of the given size
        frames   = deque(maxlen=size)
        previous = time.monotonic()
        yield 0.0  # Keep first value zero

        while True:
            now = time.monotonic()
            frames.append(now - previous)
            previous = now
            average = len(frames) / sum(frames)
            yield round(average)

    fps = average_fps(frame_rate)
    def inference_cb(input_tensor, size):
        global fps
        outputs = rknn.inference(inputs=[input_tensor])
        output = outputs[0][0]
        output_sorted = sorted(output, reverse=True)
        score = output_sorted[0]
        index = np.where(output == score)
        #print(score, index);
        stats = 'score:{:.2f}, fps:{}'.format(score, next(fps))
        label = labels[index[0][0]].capitalize()
        drawing = svgwrite.Drawing('', size=size)
        drawing.add(drawing.text(stats, insert=(11, 21), fill='black', font_size='20'))
        drawing.add(drawing.text(stats, insert=(10, 20), fill='white', font_size='20'))
        drawing.add(drawing.text(label, insert=(11, 41), fill='black', font_size='20'))
        drawing.add(drawing.text(label, insert=(10, 40), fill='white', font_size='20'))
        return drawing.tostring()
    
    try:
        inference = Inference(gst_pipeline, inference_cb, src_size)
        inference.run()
    except:
        print("Terminated.")
    finally:
        rknn.release()

