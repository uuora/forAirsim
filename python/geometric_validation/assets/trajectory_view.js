/* Offline viewer: layout, visibility and camera changes retain recorded coordinates. */
(async function(){
  'use strict';
  const views=JSON.parse(document.getElementById('trajectory-data').textContent).views;
  const plot=document.getElementById('plot');
  const available=views.flatMap(v=>v.series).filter((s,i,all)=>all.findIndex(v=>v.method===s.method)===i);
  const state={index:0,scale:'display',reference:false,arrangement:'overlay',width:6,
    methods:new Set(available.map(s=>s.method)),cameras:new Map()};
  const narrow=window.matchMedia('(max-width:620px)');
  const config={responsive:true,scrollZoom:true,displaylogo:false,
    modeBarButtonsToRemove:['orbitRotation','hoverClosest3d'],
    toImageButtonOptions:{format:'png',filename:'trajectory_3d',width:1600,height:1000,scale:1}};
  let queue=Promise.resolve(),syncingCamera=false;
  function enqueue(action){
    queue=queue.then(async()=>{
      window.trajectoryBusy=true;
      try{await action();}finally{window.trajectoryBusy=false;}
    }).catch(error=>{
      document.getElementById('coordinates').textContent='三维图加载失败：'+error.message;
      window.trajectoryError=String(error);
    });
    return queue;
  }
  function camera(view,preset='oblique'){
    const eye={oblique:{x:1.25,y:-1.6,z:view.planar?1.65:1.15},top:{x:0,y:0,z:2.3},
      front:{x:0,y:-2.3,z:.001},side:{x:2.3,y:0,z:.001}}[preset];
    return {eye,center:{x:0,y:0,z:0},up:preset==='top'?{x:0,y:1,z:0}:{x:0,y:0,z:1},projection:{type:'orthographic'}};
  }
  function sceneName(index){return index===0?'scene':'scene'+(index+1);}
  function sceneNames(){return state.arrangement==='separate'?views[state.index].series.map((s,i)=>sceneName(i)):['scene'];}
  function currentCamera(){const view=views[state.index];return state.cameras.get(view.id)||camera(view);}
  function lineWidth(method){return state.arrangement==='separate'?state.width:state.width*(method==='hermite'?.68:method==='bspline'?.83:1);}
  function waypoints(view,scene){
    const last=view.points.length-1;
    return {type:'scatter3d',scene,mode:'markers+text',x:view.points.map(p=>p[0]),y:view.points.map(p=>p[1]),z:view.points.map(p=>p[2]),
      text:view.points.map((p,i)=>i===0?'S':i===last?'E':'W'+i),textposition:'top center',
      textfont:{color:'#34434f',size:12},showlegend:false,
      marker:{size:view.points.map((p,i)=>i===0||i===last?6:4),
        color:view.points.map((p,i)=>i===0?'#32816d':i===last?'#cf4854':'#fff'),
        symbol:view.points.map((p,i)=>i===last?'diamond':i===0?'square':'circle'),line:{width:2,color:'#445764'}},
      hovertemplate:'%{text}<br>X=%{x:.3f} m<br>Y=%{y:.3f} m<br>H=%{z:.3f} m<extra></extra>',meta:{kind:'waypoint'}};
  }
  function traces(view){
    const data=[];
    // Reference is drawn underneath measured lines, including in each independent scene.
    view.series.forEach((series,index)=>{
      const scene=state.arrangement==='separate'?sceneName(index):'scene';
      ['reference','measured'].forEach(kind=>{
        const points=series[kind],measured=kind==='measured';
        data.push({type:'scatter3d',scene,mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
          name:series.label+(measured?' · 实测':' · 参考'),showlegend:false,customdata:series.t,
          meta:{method:series.method,kind},
          visible:state.methods.has(series.method)&&(measured||state.reference),
          line:{color:measured?series.color:'#737f8b',width:measured?lineWidth(series.method):2,
            dash:measured?(series.method==='hermite'&&state.arrangement==='overlay'?'longdash':'solid'):'dot'},
          opacity:measured?1:.8,
          hovertemplate:series.label+(measured?' · 实测':' · 参考')+'<br>t=%{customdata:.3f} s<br>X=%{x:.3f} m<br>Y=%{y:.3f} m<br>H=%{z:.3f} m<extra></extra>'});
      });
      if(state.arrangement==='separate')data.push(waypoints(view,scene));
    });
    if(state.arrangement==='overlay')data.push(waypoints(view,'scene'));
    return data;
  }
  function sceneLayout(view,domain){
    const axis=(title,index)=>({title:{text:title,font:{size:12,color:'#586874'}},range:view.limits[index],
      showbackground:true,backgroundcolor:index===2?'#f6f9fb':'#fcfdfe',
      showline:true,linecolor:'#b7c5cf',gridcolor:'#dce5eb',zeroline:false,
      tickfont:{size:11,color:'#667582'},nticks:4,showspikes:false});
    return {domain,xaxis:axis('X (m)',0),yaxis:axis('Y (m)',1),zaxis:axis('高度 (m)',2),
      aspectmode:state.scale==='metric'?'data':'manual',aspectratio:{x:1.55,y:1.1,z:.8},
      camera:currentCamera(),dragmode:'orbit',bgcolor:'#fff',uirevision:view.id};
  }
  function layout(view){
    const result={margin:{l:0,r:0,t:state.arrangement==='separate'?28:0,b:0},paper_bgcolor:'#fff',plot_bgcolor:'#fff',
      uirevision:view.id,hoverlabel:{font:{family:'Segoe UI, Microsoft YaHei, sans-serif',size:13}},annotations:[]};
    if(state.arrangement==='overlay')result.scene=sceneLayout(view,{x:[0,1],y:[0,1]});
    else view.series.forEach((series,i)=>{
      const n=view.series.length,gap=.025;
      const domain=narrow.matches?{x:[0,1],y:[1-(i+1)/n+gap,1-i/n-gap]}:
        {x:[i/n+gap/2,(i+1)/n-gap/2],y:[0,1]};
      result[sceneName(i)]=sceneLayout(view,domain);
      result.annotations.push({text:series.label+' · 实测',x:(domain.x[0]+domain.x[1])/2,y:domain.y[1],
        xref:'paper',yref:'paper',xanchor:'center',yanchor:'bottom',showarrow:false,font:{color:series.color,size:14}});
    });
    return result;
  }
  function publish(){
    const view=views[state.index];
    window.trajectoryState={environment:view.id,scale:state.scale,arrangement:state.arrangement,
      methods:Array.from(state.methods),reference:state.reference,width:state.width};
    document.getElementById('empty').hidden=view.series.some(s=>state.methods.has(s.method));
  }
  async function visibility(){
    await Plotly.restyle(plot,{visible:plot.data.map(trace=>trace.meta.kind==='waypoint'||
      (state.methods.has(trace.meta.method)&&(trace.meta.kind==='measured'||state.reference)))});
    publish();
  }
  async function draw(){
    plot.classList.toggle('separate',state.arrangement==='separate');
    document.querySelector('main').classList.toggle('separate-mode',state.arrangement==='separate');
    await Plotly.react(plot,traces(views[state.index]),layout(views[state.index]),config);
    await Plotly.Plots.resize(plot);publish();
  }
  async function select(index){
    state.index=index;
    document.querySelectorAll('[role=tab]').forEach((button,i)=>{
      button.setAttribute('aria-selected',String(i===index));button.tabIndex=i===index?0:-1;
    });
    document.getElementById('camera').value=state.cameras.has(views[index].id)?'custom':'oblique';
    document.getElementById('coordinates').textContent='X / Y / 高度 · m';
    await draw();
  }
  views.forEach((view,index)=>{
    const button=document.createElement('button');button.type='button';button.textContent=view.label;
    button.setAttribute('role','tab');button.setAttribute('aria-selected',String(index===0));button.tabIndex=index===0?0:-1;
    button.addEventListener('click',()=>enqueue(()=>select(index)));
    button.addEventListener('keydown',event=>{
      let next=index;if(event.key==='ArrowRight')next=(index+1)%views.length;
      else if(event.key==='ArrowLeft')next=(index+views.length-1)%views.length;
      else if(event.key==='Home')next=0;else if(event.key==='End')next=views.length-1;else return;
      event.preventDefault();enqueue(()=>select(next));document.querySelectorAll('[role=tab]')[next].focus();
    });
    document.getElementById('environments').append(button);
  });
  available.forEach(series=>{
    const label=document.createElement('label');const input=document.createElement('input');
    input.type='checkbox';input.checked=true;input.dataset.method=series.method;input.setAttribute('aria-label',series.label);
    const swatch=document.createElement('span');swatch.className='swatch '+series.method;swatch.style.setProperty('--color',series.color);
    label.append(input,swatch,document.createTextNode(series.label));document.getElementById('methods').append(label);
    input.addEventListener('change',()=>{
      const checked=input.checked;
      enqueue(async()=>{if(checked)state.methods.add(series.method);else state.methods.delete(series.method);await visibility();});
    });
  });
  document.getElementById('reference').addEventListener('change',event=>{
    const checked=event.target.checked;enqueue(async()=>{state.reference=checked;await visibility();});
  });
  document.getElementById('line-width').addEventListener('input',event=>{
    const width=Number(event.target.value);document.getElementById('line-value').textContent=String(width);
    enqueue(async()=>{
      state.width=width;
      await Plotly.restyle(plot,{'line.width':plot.data.map(t=>t.meta.kind==='measured'?lineWidth(t.meta.method):t.meta.kind==='reference'?2:0)});
      publish();
    });
  });
  document.querySelectorAll('[data-arrangement]').forEach(button=>button.addEventListener('click',()=>enqueue(async()=>{
    state.arrangement=button.dataset.arrangement;
    document.querySelectorAll('[data-arrangement]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    await draw();
  })));
  document.querySelectorAll('[data-scale]').forEach(button=>button.addEventListener('click',()=>enqueue(async()=>{
    state.scale=button.dataset.scale;
    document.querySelectorAll('[data-scale]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    document.getElementById('scale-note').textContent=state.scale==='metric'?'各轴 1 米等长':'展示轴比例 1.55 : 1.10 : 0.80';
    const update={};sceneNames().forEach(name=>{update[name+'.aspectmode']=state.scale==='metric'?'data':'manual';});
    await Plotly.relayout(plot,update);publish();
  })));
  document.getElementById('camera').addEventListener('change',event=>{
    const preset=event.target.value;if(preset==='custom')return;
    enqueue(async()=>{
      const chosen=camera(views[state.index],preset);state.cameras.set(views[state.index].id,chosen);
      const update={};sceneNames().forEach(name=>{update[name+'.camera']=chosen;});
      syncingCamera=true;try{await Plotly.relayout(plot,update);}finally{syncingCamera=false;}
    });
  });
  narrow.addEventListener('change',()=>enqueue(draw));
  await select(0);
  // Independent scenes share a camera so rotations and zoom preserve comparability.
  plot.on('plotly_relayout',event=>{
    if(syncingCamera)return;
    const key=Object.keys(event).find(k=>/^scene\d*\.camera$/.test(k));if(!key)return;
    const chosen=JSON.parse(JSON.stringify(event[key]));state.cameras.set(views[state.index].id,chosen);
    document.getElementById('camera').value='custom';
    if(state.arrangement!=='separate')return;
    enqueue(async()=>{
      const update={};sceneNames().forEach(name=>{update[name+'.camera']=chosen;});
      syncingCamera=true;try{await Plotly.relayout(plot,update);}finally{syncingCamera=false;}
    });
  });
  const reportSize=()=>{if(parent!==window)parent.postMessage({type:'trajectory-height',height:Math.ceil(document.querySelector('main').getBoundingClientRect().height)+4},'*');};
  new ResizeObserver(reportSize).observe(document.querySelector('main'));reportSize();
  plot.on('plotly_hover',event=>{
    const p=event.points[0];
    document.getElementById('coordinates').textContent=(p.data.meta.kind==='waypoint'?p.text:p.data.name+' · '+Number(p.customdata).toFixed(3)+' s')+
      ' · X '+Number(p.x).toFixed(3)+' / Y '+Number(p.y).toFixed(3)+' / 高度 '+Number(p.z).toFixed(3)+' m';
  });
  window.trajectoryReady=true;
})().catch(error=>{document.getElementById('coordinates').textContent='三维图加载失败：'+error.message;window.trajectoryError=String(error);});
