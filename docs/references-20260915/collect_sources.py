import concurrent.futures, datetime, hashlib, json, pathlib, urllib.request, urllib.parse
ROOT=pathlib.Path(__file__).resolve().parent
SOURCES=ROOT/'sources'; SOURCES.mkdir(exist_ok=True)
QUERIES=[
('airsim','AirSim High-Fidelity Visual and Physical Simulation for Autonomous Vehicles'),
('seadrones','SeaDronesSee A Maritime Benchmark for Detecting Humans in Open Water'),
('mods','MODS A USV-oriented object detection and obstacle segmentation benchmark'),
('zhang','A flexible new technique for camera calibration'),
('kalibr','Unified temporal and spatial calibration for multi-sensor systems'),
('vins','VINS-Mono A Robust and Versatile Monocular Visual-Inertial State Estimator'),
('orbslam3','ORB-SLAM3 An Accurate Open-Source Library for Visual Visual-Inertial and Multimap SLAM'),
('raftstereo','RAFT-Stereo Multilevel Recurrent Field Transforms for Stereo Matching'),
('yolov7','YOLOv7 Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors'),
('bytetrack','ByteTrack Multi-Object Tracking by Associating Every Detection Box'),
('hota','HOTA A Higher Order Metric for Evaluating Multi-object Tracking'),
('react','ReAct Synergizing Reasoning and Acting in Language Models'),
('saycan','Do As I Can Not As I Say Grounding Language in Robotic Affordances'),
('aoi','Age of Information An Introduction and Survey'),
('fresh','Real-time status How often should one update'),
('searchrescue','Unmanned aerial vehicles for maritime search and rescue')]
def fetch(key,url):
 now=datetime.datetime.now(datetime.timezone.utc).isoformat(); out={'key':key,'url':url,'retrieved_at':now}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ResearchBibliography/1.0 (bounded academic metadata retrieval)'})
  with urllib.request.urlopen(req,timeout=35) as r: b=r.read();out.update(status=r.status,final_url=r.url)
  suffix='.pdf' if b.startswith(b'%PDF-') else ('.json' if b.lstrip().startswith(b'{') else '.html')
  p=SOURCES/(key+suffix); p.write_bytes(b);out.update(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
 except Exception as e: out.update(status='unavailable',error=repr(e))
 return out
if __name__=='__main__':
 jobs=[(k,'https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':q,'rows':3})) for k,q in QUERIES]
 jobs += [('lmucs_github','https://api.github.com/repos/wp19991/LMUCS/readme'),('airsim_arxiv','https://arxiv.org/abs/1705.05065'),('react_arxiv','https://arxiv.org/abs/2210.03629'),('saycan_arxiv','https://arxiv.org/abs/2204.01691')]
 logs=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futs=[pool.submit(fetch,k,u) for k,u in jobs]
  for f in concurrent.futures.as_completed(futs):
   row=f.result(); logs.append(row);print(row['key'],row['status'],flush=True)
 (ROOT/'search_log.json').write_text(json.dumps({'search_queries':dict(QUERIES),'requests':logs},ensure_ascii=False,indent=2),encoding='utf-8')
