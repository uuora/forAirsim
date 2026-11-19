const PANEL_HEIGHT = 312;
const PAD = {left:47, right:13, top:36, bottom:23};
const PANELS = [
  {title:'机体欧拉角', fields:[
    {key:'roll', name:'滚转', color:'#0b6d71'},
    {key:'pitch', name:'俯仰', color:'#5b42a6'},
    {key:'yaw', name:'偏航', color:'#a84f10'},
  ], ticks:[-180,-90,0,90,180]},
  {title:'姿态误差和倾角', fields:[
    {key:'attitudeError', name:'姿态误差', color:'#b13536'},
    {key:'tilt', name:'实际倾角', color:'#235da1'},
  ], ticks:[0,45,90,135,180]},
];

export class AttitudeChart {
  constructor(canvas, onSeek) {
    this.canvas = canvas;
    this.base = document.createElement('canvas');
    this.series = [];
    this.selectedTime = null;
    this.limits = [];
    this.events = [];
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('pointerdown', event => {
      if (!this.series.length) return;
      const bounds = canvas.getBoundingClientRect();
      const fraction = (event.clientX - bounds.left - PAD.left) / (bounds.width - PAD.left - PAD.right);
      onSeek(this.start + this.span * Math.max(0, Math.min(1, fraction)));
    });
  }

  setSeries(series, limits, events) {
    this.series = series;
    this.selectedTime = null;
    this.start = series[0]?.time ?? 0;
    this.span = Math.max(.02, (series.at(-1)?.time ?? 1) - this.start);
    this.limits = limits;
    this.events = events;
    this.draw();
  }

  paint() {
    if (!this.width) return;
    const context = this.canvas.getContext('2d');
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.drawImage(this.base, 0, 0);
    if (!Number.isFinite(this.selectedTime)) return;
    context.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
    context.strokeStyle = '#172e3c';
    context.lineWidth = 1;
    const x = PAD.left + (this.selectedTime-this.start)*this.plotWidth/this.span;
    context.beginPath();
    context.moveTo(x,PAD.top);
    context.lineTo(x,PANEL_HEIGHT-PAD.bottom);
    context.stroke();
  }

  setTime(time) {
    this.selectedTime = time;
    this.paint();
  }

  draw() {
    this.width = Math.floor(this.canvas.getBoundingClientRect().width);
    if (this.width < 90) return;
    this.pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    const w = this.width, dpr = this.pixelRatio;
    const plotWidth = w-PAD.left-PAD.right;
    this.plotWidth = plotWidth;
    this.canvas.width = this.base.width = Math.round(w*dpr);
    this.canvas.height = this.base.height = Math.round(PANEL_HEIGHT*dpr);
    const context = this.base.getContext('2d');
    context.scale(dpr,dpr);
    context.font = '12px system-ui, sans-serif';
    context.fillStyle = '#f7fafc';
    context.fillRect(0,0,w,PANEL_HEIGHT);
    if (!this.series.length) {
      context.fillStyle = '#4b6576';
      context.fillText('此记录暂无可回放的有效姿态样本。', 12, 27);
      this.paint();
      return;
    }
    const panelSize = PANEL_HEIGHT / 2;
    const plotHeight = panelSize-PAD.top-PAD.bottom;
    const timeX = value => PAD.left + plotWidth*(value-this.start)/this.span;
    PANELS.forEach((panel, index) => {
      const top = index*panelSize + PAD.top, bottom = top + plotHeight;
      const min = panel.ticks[0], max = panel.ticks.at(-1);
      const y = value => bottom-plotHeight*(value-min)/(max-min);
      context.fillStyle = '#1b3549';
      context.fillText(panel.title, PAD.left, top-21);
      let labelX = PAD.left;
      for (const field of panel.fields) {
        context.fillStyle = field.color;
        context.fillRect(labelX,top-13,12,2);
        context.fillText(field.name,labelX+16,top-8);
        labelX += 29 + context.measureText(field.name).width;
      }
      for (const tick of panel.ticks) {
        const height = y(tick);
        context.strokeStyle = '#dbe6eb';
        context.beginPath();
        context.moveTo(PAD.left,height);
        context.lineTo(w-PAD.right,height);
        context.stroke();
        context.fillStyle = '#4c6579';
        context.fillText(String(tick)+'°',6,height+3);
      }
      const tickCount = w < 430 ? 3 : 6;
      for (let tick=0; tick<=tickCount; tick++) {
        const t = this.start+this.span*tick/tickCount;
        const x = timeX(t);
        context.strokeStyle = '#dbe6eb';
        context.beginPath();
        context.moveTo(x,top);
        context.lineTo(x,bottom);
        context.stroke();
        if (index === 1) {
          context.fillStyle = '#4c6579';
          const label = t.toFixed(1)+' s';
          context.fillText(label,Math.min(x-context.measureText(label).width/2,w-context.measureText(label).width-2),bottom+17);
        }
      }
      context.save();
      context.beginPath();
      context.rect(PAD.left,top,plotWidth,plotHeight);
      context.clip();
      context.setLineDash([4,3]);
      for (const event of this.events) {
        if (!Number.isFinite(event)) continue;
        const x = timeX(event);
        context.strokeStyle = '#919ea5';
        context.beginPath();
        context.moveTo(x,top);
        context.lineTo(x,bottom);
        context.stroke();
      }
      if (index === 1) {
        for (let i=0; i<this.limits.length; i++) {
          const threshold = y(this.limits[i]);
          context.strokeStyle = panel.fields[i].color;
          context.beginPath();
          context.moveTo(PAD.left,threshold);
          context.lineTo(w-PAD.right,threshold);
          context.stroke();
        }
      }
      context.setLineDash([]);
      for (const field of panel.fields) {
        context.strokeStyle = field.color;
        context.lineWidth = 1.7;
        context.beginPath();
        let previous = null;
        for (const sample of this.series) {
          const value = sample[field.key];
          if (!Number.isFinite(value)) {
            previous = null;
            continue;
          }
          const x = timeX(sample.time), height = y(value);
          if (previous === null || (index === 0 && Math.abs(value-previous) > 180)) context.moveTo(x,height);
          else context.lineTo(x,height);
          previous = value;
        }
        context.stroke();
      }
      context.restore();
    });
    this.paint();
  }
}
