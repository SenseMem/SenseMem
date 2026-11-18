// AudioWorklet uses the browser render clock, independently of UI scheduling.
class PcmInput extends AudioWorkletProcessor {
  constructor() {
    super();
    this.packet = new ArrayBuffer(480 * 2);
    this.view = new DataView(this.packet);
    this.position = 0;
    this.inputIndex = this.nextSample = this.previous = 0;
    this.step = sampleRate / 24_000;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let index = 0; index < channels[0].length; index++) {
      let current = 0;
      for (const channel of channels) current += channel[index] / channels.length;
      while (this.nextSample <= this.inputIndex) {
        const fraction = this.inputIndex === 0 ? 1 : this.nextSample - (this.inputIndex - 1);
        const value = Math.max(-1, Math.min(1, this.previous + (current - this.previous) * fraction));
        this.view.setInt16(this.position * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
        this.position++;
        this.nextSample += this.step;
        if (this.position === 480) {
          this.port.postMessage(this.packet, [this.packet]);
          this.packet = new ArrayBuffer(480 * 2);
          this.view = new DataView(this.packet);
          this.position = 0;
        }
      }
      this.previous = current;
      this.inputIndex++;
    }
    return true;
  }
}

registerProcessor("sensemem-pcm-input", PcmInput);
