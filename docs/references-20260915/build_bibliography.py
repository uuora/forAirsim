import pathlib,json,re,html,difflib,collections
from html.parser import HTMLParser
ROOT=pathlib.Path(__file__).resolve().parent;S=ROOT/'sources'
class Meta(HTMLParser):
 def __init__(self,t):super().__init__();self.m=collections.defaultdict(list);self.feed(t)
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='meta' and 'name' in a:self.m[a['name']].append(a.get('content',''))
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
def clean(s):return html.unescape(re.sub('<[^>]+>',' ',s)).strip()
ANN={
'kalibr':('传感器与几何',1,'第2章传感器模型；第3章采集验收','先精读时间偏移与空间外参的定义，列出两相机及 IMU 的时间基准、坐标变换。','题名/元数据已核验；本轮未获得摘要，具体标定模型与公式待读原文。不能证明两台独立无人机具有恒定外参。'),
'zhang':('传感器与几何',2,'第2章相机内参与畸变；第3章标定','将相机内参、畸变参数与标定验收写入传感器清单。','本轮仅核验元数据，须读原文再引用具体算法公式；单相机内参标定不能代替跨机时间同步。'),
'seadrones':('海事任务与数据集',3,'第1章任务背景；第3章数据划分与评测','参考海上人员检测/跟踪任务，以及高度、俯视角等采集变量。','真实海上数据集的基线不能证明本项目仿真图像已具备同等难度；本文不支持由船/气球检测直接推出人员搜救性能。'),
'raftstereo':('传感器与几何',4,'第2章双目深度；第3章深度基线','选作完成几何校正后的双目匹配候选基线，后续检查共同视野和视差质量。','摘要明确面向 rectified stereo；不能把任意不同时间、不同姿态的两机图像直接输入并宣称得到可靠尺度深度。'),
'airsim':('海事任务与数据集',5,'第1章仿真相关工作；第3章平台','说明仿真平台选择与数据采集架构，区分平台能力与本项目已实测能力。','原论文仿真验证不能证明当前海面、相机噪声和标注真实性，也不能替代实机验证。'),
'mods':('海事任务与数据集',6,'第2章海事感知；第3章船端评价','参考船载障碍检测/分割及立体图像与 IMU 同步数据组织。','USV 视角与无人机俯视视角不同；障碍分割性能不等同落水人员定位或救援成功率。'),
'vins':('传感器与几何',7,'第2章位姿估计；第4章实机传感器','理解相机与 IMU 融合、尺度位姿、初始化和外参对定位链路的要求。','相机位姿估计不同于被观测目标的绝对位置；不能用系统自定位精度替代目标定位误差。'),
'orbslam3':('传感器与几何',8,'第2章视觉定位；第4章迁移候选','比较单目、双目、视觉惯性配置，作为后续真实位姿来源的候选。','原文报告来自特定基准；海面弱纹理、动态目标与本项目双机性能尚未验证。'),
'yolov7':('检测与跟踪',9,'第2章目标检测；第3章检测对照','参考检测准确性、速度与训练策略的对照组织方式；工程模型可另选。','不是 YOLOv11 的论文；不能用其性能或结构解释当前 YOLOv11 权重，也不能声称旧模型是当前最优。'),
'bytetrack':('检测与跟踪',10,'第2章跟踪；第3章遮挡实验','参考低分检测关联与轨迹恢复，设置遮挡条件下的跟踪基线。','MOT17 等数据集结果不等于海事 UAV 结果；两个视角的目标身份关联仍需额外设计与评估。'),
'hota':('检测与跟踪',11,'第3章跟踪评价指标','阅读检测、关联与定位误差分解，用于解释 ID 切换和丢失。','HOTA 不是三维定位精度或任务成功率，应与定位误差、时延等指标分开报告。'),
'aoi':('信息时效与通信',12,'第2章信息时效；第3章通信延迟实验','定义报告生成时刻、船端接收时刻与信息年龄，整理 AoI 与估计/控制的关系。','AoI 小不必然代表位置准确；本项目仍须联合记录位置误差与报告年龄，不能直接宣称时效策略创新成立。'),
'fresh':('信息时效与通信',13,'第2章 AoI 理论背景','追溯状态更新频率问题的原始来源，精读后确认指标及队列假设。','本轮仅元数据，未核验公式或实验；不能把特定更新模型结论直接推广到所有海事链路。'),
'searchrescue':('信息时效与通信',14,'第1章海上搜救通信；第3章延迟设置','参考海上 UAV 搜救通信的应用层延迟/可靠性评价。','摘要中的低时延来自 LTE 实验室软件无线电设置；不能移植为本项目 AirSim 或真实海域的实测延迟。'),
'lmucs':('轻量 Agent',15,'第2章 LLM 无人机系统；第3章指令基线','参考仓库的指令解析、检测、单目深度模块划分；核对指令准确率和边缘延迟评价。','论文元数据与仓库分别核验，未读论文正文；README 的 MiDaS 是单目深度，不能支持双机双目创新或本项目实机性能。'),
'react':('轻量 Agent',16,'第2章 Agent 工作流','参考推理与行动交替、根据外部反馈更新计划的任务组织方式。','语言/交互任务上的收益不证明 UAV 闭环控制安全、实时或更高救援成功率。'),
'saycan':('轻量 Agent',17,'第2章语言到可执行技能','参考用可执行技能与价值函数约束语言模型的动作选择。','移动操作机器人实验不能直接迁移为海空协同证据；本轮使用 arXiv 版本，不冒称已核验会议版本。')}
entries=[];reject=[]
for key,(group,order,chapter,use,limit) in ANN.items():
 sourcekey='lmucs_crossref' if key=='lmucs' else key
 if key in ['react','saycan']:
  m=Meta((S/(key+'_arxiv.html')).read_text(encoding='utf-8')).m
  e={'key':key,'title':m['citation_title'][0],'authors':[{'family':a.split(',')[0],'given':a.split(',',1)[1].strip() if ',' in a else ''} for a in m['citation_author']], 'year':int(m['citation_date'][0][:4]),'arxiv_id':m['citation_arxiv_id'][0],'venue':'arXiv','publication_type':'preprint','doi':None,'verification_status':'ARXIV_METADATA_VERIFIED','metadata_source':'sources/'+key+'_arxiv.html'}
  e['abstract']=m['citation_abstract'][0]; e['read_scope']='abstract_only';e['abstract_source']=e['metadata_source'];e['url']='https://arxiv.org/abs/'+e['arxiv_id']
 else:
  raw=json.loads((S/(sourcekey+'.json')).read_text())['message']['items'][0]
  s2=json.loads((S/(key+'_s2.json')).read_text())
  sim=difflib.SequenceMatcher(None,norm(raw['title'][0]),norm(s2['title'])).ratio()
  assert sim>.9,(key,sim)
  e={'key':key,'title':raw['title'][0],'authors':[{'given':a.get('given',''),'family':a.get('family','')} for a in raw['author']], 'year':raw.get('published-print',raw['published'])['date-parts'][0][0],'doi':raw['DOI'],'venue':raw['container-title'][-1],'publication_type':raw['type'],'volume':raw.get('volume'),'issue':raw.get('issue'),'pages':raw.get('page'),'metadata_source':'sources/'+sourcekey+'.json','semantic_scholar_id':s2['paperId'],'s2_title_similarity':sim,'verification_status':'CROSSREF_AND_S2_METADATA_MATCH','abstract':s2.get('abstract'),'arxiv_id':s2.get('externalIds',{}).get('ArXiv'),'published_online':raw.get('published-online'),'published_print':raw.get('published-print'),'crossref_updates':{k:raw[k] for k in ['updated-by','update-to'] if k in raw}}
  e['url']='https://doi.org/'+e['doi'];e['abstract_source']='sources/'+key+'_s2.json' if e['abstract'] else None
  f=S/(key+'_arxiv.html')
  if f.exists():
   m=Meta(f.read_text(encoding='utf-8')).m
   if m.get('citation_title') and difflib.SequenceMatcher(None,norm(e['title']),norm(m['citation_title'][0])).ratio()>.85:
    e['abstract']=m['citation_abstract'][0];e['abstract_source']='sources/'+key+'_arxiv.html'
   else:reject.append({'path':str(f.relative_to(ROOT)),'reason':'TITLE_MISMATCH; excluded from evidence','retrieved_title':m.get('citation_title')})
  e['read_scope']='abstract_only' if e['abstract'] else 'unknown'
 e.update(category=group,reading_order=order,chapter_use=chapter,reading_task=use,cannot_support=limit,full_text_read=False,human_read_attestation=False,method_quality='NOT_ASSESSED_FULL_TEXT',coi='NOT_CHECKED',retraction_status='NOT_ASSESSED',source_acquired=True,source_verified_against_original=False)
 if key=='lmucs':e['publication_timing_note']='Crossref print issue 2026-10 is later than search date 2026-09-15; DOI registered 2026-06-22 and resolved. Do not infer exact online-publication date. Content notes derive only from repository README.'
 if key=='airsim':e['publication_timing_note']='Proceedings print year 2018; online and arXiv first publication 2017. BibTeX uses proceedings print version.'
 if key=='hota':e['publication_timing_note']='Print year 2021; online publication 2020. BibTeX uses journal issue version.'
 entries.append(e)
entries.sort(key=lambda e:e['reading_order'])
(ROOT/'verified_metadata.json').write_text(json.dumps({'schema':'bounded-bibliography-v1','last_searched_at':'2026-09-15','scope':'targeted seed-based bibliography, not systematic review','entries':entries,'rejected_candidate_links':reject},ensure_ascii=False,indent=2),encoding='utf-8')
def esc(s):return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
bib=[]
for e in entries:
 typ='misc' if e['publication_type']=='preprint' else ('article' if e['publication_type']=='journal-article' else 'inproceedings')
 f={'title':'{'+e['title']+'}','author':' and '.join(a['family']+', '+a['given'] for a in e['authors']),'year':e['year'],'url':e['url']}
 if typ=='misc':f.update(eprint=e['arxiv_id'],archivePrefix='arXiv')
 else:f['journal' if typ=='article' else 'booktitle']=e['venue'];f['doi']=e['doi']
 for k,v in [('volume',e.get('volume')),('number',e.get('issue')),('pages',e.get('pages'))]:
  if v:f[k]=v
 if e.get('publication_timing_note'):f['note']=e['publication_timing_note']
 bib.append('@'+typ+'{'+e['key']+str(e['year'])+',\n'+',\n'.join('  '+k+' = {'+esc(v)+'}' for k,v in f.items())+'\n}')
(ROOT/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
lines=['# 论文参考文献：首批 17 篇与阅读任务','', '检索日期：2026-09-15。服务于海事搜救、无人机观察与船端报告这一当前主线。优先先解决传感器：阅读顺序 1–4；其他材料分阶段阅读。此目录是有界文献收集，不是系统综述，也不是论文创新性已成立的证明。','', '## 先读三篇','', '1. **Furgale 等（2013）多传感器时空标定**：先画清时间基准与坐标系；本轮只有书目信息，需打开原文精读。','2. **Zhang（2000）相机标定**：明确内参、畸变与验收项目；本轮只有书目信息。','3. **SeaDronesSee（2022）**：把海上人员检测任务与采集高度/视角对应起来；摘要已读，提供公开 PDF。','', '## 文献矩阵','', '|顺序|论文与年份|用途/章节|可借鉴内容|不能据此声称|本轮阅读范围|','|---|---|---|---|---|---|']
for e in entries:
 scope='摘要' if e['read_scope']=='abstract_only' else ('书目 + 仓库 README（非论文正文）' if e['key']=='lmucs' else '仅书目，待读原文')
 lines.append('|'+ '|'.join([str(e['reading_order']),'['+e['title']+']('+e['url']+') ('+str(e['year'])+')',e['chapter_use'],e['reading_task'],e['cannot_support'],scope])+'|')
lines+=['','## 逐篇引用与核验记录','']
for e in entries:
 authors=[]
 for a in e['authors']:
  initials=' '.join(x[0]+'.' for x in a['given'].replace('-',' ').split() if x)
  authors.append(a['family']+', '+initials)
 if len(authors)>20:authors=authors[:19]+['…',authors[-1]]
 names=', '.join(authors[:-1])+', & '+authors[-1] if len(authors)>1 else authors[0]
 lines += ['### '+str(e['reading_order'])+'. '+e['key'], '',f"{names} ({e['year']}). {e['title']}. *{e['venue']}*. {e['url']}",'',f"- 书目状态：{e['verification_status']}；原始响应：`{e['metadata_source']}`。",f"- 用途：{e['reading_task']}",f"- 引用边界：{e['cannot_support']}",'- 方法质量：未作全文方法审查；作者资历、利益冲突、撤稿状态未完成核查。']
 if e.get('publication_timing_note'):lines.append('- 日期说明：'+e['publication_timing_note'])
 if e.get('arxiv_id'):lines.append('- 开放预印本入口：https://arxiv.org/abs/'+e['arxiv_id']+' （与正式版版本可能不同，引用前核对。）')
 lines.append('')
lines += ['## LMUCS 单独说明','', '正式期刊记录题名为 “LMUCS: Lightweight LLM-driven UAV control system with multimodal perception for autonomous material search and localization”，DOI 10.1016/j.ast.2026.112949；与 GitHub README 的题名相符，Crossref 与 Semantic Scholar 作者信息可追溯。期次为 2026 年 10 月，晚于检索日；DOI 登记日期为 2026 年 6 月 22 日，且当前能够解析。这支持“期刊书目记录已存在”，不等于已读正式全文。','', 'Crossref 还返回 SSRN 10.2139/ssrn.5397714，题名末尾为 “material deliver”，作者排列也有差异；本轮未将其合并或作为第二篇独立研究计数。README 的轻量模型、YOLOv11n、MiDaS 等描述属于软件项目自述，不能替代论文实验审查。仓库快照见 `sources/lmucs_readme.md`。','', '## 文件使用与限制','', '- `references.bib`：17 条 BibTeX，可导入 Zotero/文献管理工具；保留正式版本与预印本版本的区分。','- `verified_metadata.json`：核验来源、阅读范围、作者、年份、DOI/arXiv ID 与未完成检查。','- `sources/`：实际 HTTP 响应快照与公开 PDF（成功下载者）；有文件不代表已阅读全文。','- `SEARCH_STRATEGY.md`：检索策略、失败/误匹配与覆盖限制；`*_log.json` 保存实际请求记录。','', '**AI 使用披露：** 本文献清单由 AI 辅助检索、元数据比对和阅读任务整理。13 篇读取摘要，4 篇仅书目（其中 LMUCS 另读仓库 README）；未执行任何论文全文精读。应用建议是拟定阅读/实验任务，不是作者原文逐句结论。未开展系统综述、独立专家质量评定或新颖性穷尽检索。']
count=sum(e['read_scope']=='abstract_only' for e in entries)
lines=[l.replace('13 篇读取摘要，4 篇仅书目',f'{count} 篇读取摘要，{len(entries)-count} 篇仅书目') for l in lines]
(ROOT/'阅读指南与文献矩阵.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('entries',len(entries),'abstracts',count,'mismatch candidates',reject)
