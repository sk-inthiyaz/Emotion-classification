class AudioVisualizer {
    constructor(canvasId, color = '#2563eb') {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.color = color;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.source = null;
        this.animationId = null;
        this.isActive = false;

        // Resize canvas to match display size
        this.resize();
        this.audioContext = null;
        this.style = 'wave'; // 'wave' (time-domain) or 'bars' (frequency)
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    async init(streamOrElement) {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        // Ensure canvas is correctly scaled for devicePixelRatio
        const rect = this.canvas.getBoundingClientRect();
        const ratio = Math.max(1, Math.floor(window.devicePixelRatio || 1));
        this.canvas.width = Math.floor(rect.width * ratio);
        this.canvas.height = Math.floor(rect.height * ratio);
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(ratio, ratio);

        try {
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume().catch(() => {});
                console.debug('[Visualizer] AudioContext resumed');
            }
        } catch (_) {}

        if (this.source) {
            try {
                this.source.disconnect();
            } catch (_) {}
        }

        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 1024;
        const bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(bufferLength);

        if (streamOrElement instanceof MediaStream) {
            this.source = this.audioContext.createMediaStreamSource(streamOrElement);
            this.timeDomainArray = new Uint8Array(this.analyser.fftSize);
        } else if (streamOrElement instanceof HTMLAudioElement) {
            // Create MediaElementSource only once per element
            if (!streamOrElement._source) {
                streamOrElement._source = this.audioContext.createMediaElementSource(streamOrElement);
            }
            this.source = streamOrElement._source;
            this.analyser.fftSize = 2048; // smoother waveform for element playback
            this.timeDomainArray = new Uint8Array(this.analyser.fftSize);
        }

        this.source.connect(this.analyser);
        if (streamOrElement instanceof HTMLAudioElement) {
            // Only connect analyser to destination (optional) if needed
            try { this.analyser.connect(this.audioContext.destination); } catch (_) {}
        }

        this.isActive = true;
        console.debug('[Visualizer] init with', streamOrElement instanceof MediaStream ? 'MediaStream' : 'HTMLAudioElement');
        this.draw();
    }

    draw() {
        if (!this.isActive || !this.analyser) return;

        this.animationId = requestAnimationFrame(() => this.draw());

        const width = this.canvas.width / window.devicePixelRatio;
        const height = this.canvas.height / window.devicePixelRatio;
        this.ctx.clearRect(0, 0, width, height);

        if (this.style === 'wave') {
            if (!this.timeDomainArray || this.timeDomainArray.length !== this.analyser.fftSize) {
                this.timeDomainArray = new Uint8Array(this.analyser.fftSize);
            }
            this.analyser.getByteTimeDomainData(this.timeDomainArray);

            const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
            gradient.addColorStop(0, '#22d3ee');
            gradient.addColorStop(1, this.color);
            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = 2;

            this.ctx.beginPath();
            const sliceWidth = width / this.timeDomainArray.length;
            let x = 0;
            for (let i = 0; i < this.timeDomainArray.length; i++) {
                const v = this.timeDomainArray[i] / 128.0;
                const y = (v * height) / 2;
                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
                x += sliceWidth;
            }
            this.ctx.stroke();

            // Subtle center glow line for depth
            this.ctx.globalAlpha = 0.08;
            this.ctx.fillStyle = '#60a5fa';
            this.ctx.fillRect(0, height / 2 - 2, width, 4);
            this.ctx.globalAlpha = 1;
        } else {
            // Frequency bars
            this.analyser.getByteFrequencyData(this.dataArray);
            const barWidth = Math.max(1, (width / this.dataArray.length) * 2.2);
            let x = 0;
            for (let i = 0; i < this.dataArray.length; i++) {
                const barHeight = (this.dataArray[i] / 255) * height * 1.2;
                const g = this.ctx.createLinearGradient(0, height, 0, height - barHeight);
                g.addColorStop(0, '#22d3ee');
                g.addColorStop(0.5, this.color);
                g.addColorStop(1, '#a855f7');
                this.ctx.fillStyle = g;
                if (this.ctx.roundRect) {
                    this.ctx.beginPath();
                    this.ctx.roundRect(x, height - barHeight, barWidth - 2, barHeight, 4);
                    this.ctx.fill();
                } else {
                    this.ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
                }
                x += barWidth + 1;
            }
        }
    }

    stop() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.ctx) {
            const width = this.canvas.width / window.devicePixelRatio;
            const height = this.canvas.height / window.devicePixelRatio;
            this.ctx.clearRect(0, 0, width, height);
        }
        try {
            if (this.source && this.analyser) {
                // Disconnect to avoid lingering audio graph connections
                this.source.disconnect();
                this.analyser.disconnect();
            }
        } catch (_) {}
        console.debug('[Visualizer] stopped');
    }
}

// Initialize visualizers when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.recordingVisualizer = new AudioVisualizer('recordingCanvas', '#ef4444'); // Red for recording
    window.recordingVisualizer.style = 'wave'; // smooth waveform for recording

    window.playbackVisualizer = new AudioVisualizer('playbackCanvas', '#2563eb'); // Blue for playback
    window.playbackVisualizer.style = 'bars'; // frequency bars for uploaded playback
});
