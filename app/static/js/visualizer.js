class AudioVisualizer {
    constructor(canvasId, audioSource) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(this.bufferLength);
        this.isActive = false;

        // Handle window resize
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        if (this.canvas) {
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        }
    }

    connectStream(stream) {
        if (this.source) {
            this.source.disconnect();
        }
        this.source = this.audioContext.createMediaStreamSource(stream);
        this.source.connect(this.analyser);
        this.isActive = true;
        this.draw();
    }

    connectElement(audioElement) {
        if (this.source) {
            this.source.disconnect();
        }
        // Create source only once per element to avoid errors
        if (!audioElement._source) {
            audioElement._source = this.audioContext.createMediaElementSource(audioElement);
        }
        this.source = audioElement._source;
        this.source.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);
        this.isActive = true;
        this.draw();
    }

    draw() {
        if (!this.isActive) return;

        requestAnimationFrame(() => this.draw());

        this.analyser.getByteFrequencyData(this.dataArray);

        const width = this.canvas.width;
        const height = this.canvas.height;
        const barWidth = (width / this.bufferLength) * 2.5;
        let barHeight;
        let x = 0;

        this.ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < this.bufferLength; i++) {
            barHeight = this.dataArray[i] / 2;

            // Gradient fill
            const gradient = this.ctx.createLinearGradient(0, height - barHeight, 0, height);
            gradient.addColorStop(0, '#4f46e5'); // Primary
            gradient.addColorStop(1, '#818cf8'); // Lighter

            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(x, height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
        }
    }
}

// Initialize globally
window.AudioVisualizer = AudioVisualizer;
