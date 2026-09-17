"""Run after build_bibliography.py; align the first batch to the latest scope document."""
import pathlib,json,re,html,collections,difflib
from html.parser import HTMLParser
P=pathlib.Path(__file__).resolve().parent;S=P/'sources'
class Meta(HTMLParser):
 def __init__(self,t):super().__init__();self.m=collections.defaultdict(list);self.feed(t)
 def handle_starttag(self,t,a):
  a=dict(a)
  if t=='meta' and 'name' in a:self.m[a['name']].append(a.get('content',''))
def norm(s):return re.sub('[^a-z0-9]','',s.lower())
extra={
'adaptive':('自适应检索',1,'第2章最近邻方法；第3章复杂度路由基线','按问题复杂度在无检索、单次与迭代检索间选择，用作必须比较的最近邻基线。','换成海事语料本身不能构成创新；须比较观测时效/冲突信号是否带来额外价值。'),
'rag':('检索增强基础',2,'第2章 RAG；第3章单次检索基线','理解参数知识与非参数知识库的结合，确定检索证据与生成输出的关系。','原始 RAG 训练体系不是任意本地脚本的直接复现；不能用其问答成绩证明海事报告正确。'),
'selfrag':('自适应检索',3,'第2章检索决策/证据评价；第3章对照设计','参考按需检索与反思 token 对检索内容和生成内容的检查机制。','论文的 7B/13B 实验不能直接作为 0.5B—1.5B 模型的效果或显存保证；简单提示检查不等于复现 Self-RAG。'),
'lora':('轻量模型训练',5,'第2章参数高效微调；第3章训练设置','参考冻结基础权重、训练低秩适配参数的方式，将微调与检索增益分开实验。','不能由可训练参数少推断 RTX 3070 8GB 一定可运行任意上下文/批量；实际显存与延迟必须测量。'),
'qwen':('轻量模型训练',6,'第2章基础模型；第3章模型选择','从模型系列技术报告追溯候选模型与训练背景，再核对具体 checkpoint 的模型卡和许可证。','技术报告是作者自述的预印本；不能把系列最大模型成绩归到 0.5B/1.5B，也不保证当前硬件微调成功。'),
'latentrag':('检索效率',9,'第2章近期 Agentic RAG；第4章延迟讨论','参考多轮自然语言推理/子查询生成导致开销的分析，以及潜在空间检索方案。','2026 预印本尚未核验同行评审；摘要提速数字不能外推至当前模型、任务或硬件，也不要求本月实现该框架。')}
data=json.loads((P/'verified_metadata.json').read_text(encoding='utf-8'))
remove={'vins','orbslam3','yolov7','fresh','searchrescue'}
excluded=[e for e in data['entries'] if e['key'] in remove]
if excluded:(P/'scope_deferred_metadata.json').write_text(json.dumps({'reason':'当前论文主线为轻量语言智能体与自适应RAG，以下仅保留检索记录，不计首批参考条目','entries':excluded},ensure_ascii=False,indent=2),encoding='utf-8')
es=[e for e in data['entries'] if e['key'] not in remove and e['key'] not in extra]
for key,(cat,order,chapter,use,limit) in extra.items():
 if key=='adaptive':
  raw=json.loads((S/'adaptive_crossref.json').read_text())['message'];m=Meta((S/'adaptive_acl.html').read_text()).m
  text=(S/'adaptive_acl.html').read_text();match=re.search(r'<div[^>]*class="?acl-abstract[^>]*>(.*?)</div>',text,re.S)
  abstract=html.unescape(re.sub('<[^>]+>',' ',match.group(1))).strip() if match else None
  if not abstract:
   start=text.find('Retrieval-Augmented Large Language Models (LLMs)');end=text.find('Anthology ID:',start);abstract=html.unescape(re.sub('<[^>]+>',' ',text[start:end])).strip()
  e={'key':key,'title':raw['title'][0],'authors':[{'given':a.get('given',''),'family':a['family']} for a in raw['author']],'year':2024,'doi':raw['DOI'],'arxiv_id':None,'venue':raw['container-title'][0],'pages':raw.get('page'),'publication_type':'proceedings-article','url':'https://aclanthology.org/2024.naacl-long.389/','metadata_source':'sources/adaptive_crossref.json','abstract':abstract,'abstract_source':'sources/adaptive_acl.html','verification_status':'ACL_AND_CROSSREF_METADATA_VERIFIED'}
 else:
  m=Meta((S/(key+'_arxiv.html')).read_text()).m
  e={'key':key,'title':m['citation_title'][0],'authors':[{'family':a.split(',')[0],'given':a.split(',',1)[1].strip() if ',' in a else ''} for a in m['citation_author']],'year':int(m['citation_date'][0][:4]),'arxiv_id':m['citation_arxiv_id'][0],'doi':None,'venue':'arXiv','publication_type':'preprint','url':'https://arxiv.org/abs/'+m['citation_arxiv_id'][0],'metadata_source':'sources/'+key+'_arxiv.html','abstract':m['citation_abstract'][0],'abstract_source':'sources/'+key+'_arxiv.html','verification_status':'ARXIV_METADATA_VERIFIED'}
 sf=S/(key+'_s2.json')
 if sf.exists():
  sr=json.loads(sf.read_text());sim=difflib.SequenceMatcher(None,norm(e['title']),norm(sr['title'])).ratio();assert sim>.9,(key,sim)
  e.update(semantic_scholar_id=sr['paperId'],s2_title_similarity=sim)
 e['authors']=[a for a in e['authors'] if any(c.isalnum() for c in a['family'])]
 if key=='qwen':e['metadata_cleanup_note']='arXiv citation_author has a standalone colon separator; removed punctuation-only entry, retained Qwen group and named authors.'
 e.update(category=cat,reading_order=order,chapter_use=chapter,reading_task=use,cannot_support=limit,full_text_read=False,human_read_attestation=False,method_quality='NOT_ASSESSED_FULL_TEXT',coi='NOT_CHECKED',retraction_status='NOT_ASSESSED',source_acquired=True,source_verified_against_original=False,read_scope='abstract_only',research_role='主线')
 es.append(e)
roles={'lmucs':('主线',4),'react':('主线',7),'saycan':('主线',8),'aoi':('主线',10),'seadrones':('支撑',11),'airsim':('支撑',12),'mods':('支撑',13),'kalibr':('支撑',14),'zhang':('支撑',15),'raftstereo':('扩展',16),'bytetrack':('扩展',17),'hota':('扩展',18)}
for e in es:
 if e['key'] in roles:e['research_role'],e['reading_order']=roles[e['key']]
 if e['key']=='lmucs':e['chapter_use']='第2章轻量无人机语言系统；第3章指令理解与延迟基线'
 if e['key']=='aoi':e['chapter_use']='第2章观测时效；第3章过期证据识别标签'
 if e['key'] in ['seadrones','airsim','mods','kalibr','zhang']:e['chapter_use']='第2章观测来源基础；第4章实验数据与传感器记录'
 if e['research_role']=='扩展':e['chapter_use']='可选感知扩展/展望，非当前语言主线必做'
es.sort(key=lambda e:e['reading_order']);data.update(entries=es,scope_authority='D:/forAirsim/docs/海上搜救Agent_平台与研究范围决策.md',scope='Lightweight language agent and adaptive RAG for maritime SAR; two UAVs are observation sources, not assumed stereo')
(P/'verified_metadata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def esc(v):return str(v).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
bib=[]
for e in es:
 typ='misc' if e['publication_type']=='preprint' else ('article' if e['publication_type']=='journal-article' else 'inproceedings')
 f={'title':'{'+e['title']+'}','author':' and '.join(a['family']+(', '+a['given'] if a['given'] else '') for a in e['authors']),'year':e['year'],'url':e['url']}
 if typ=='misc':f.update(eprint=e['arxiv_id'],archivePrefix='arXiv')
 else:f.update({('journal' if typ=='article' else 'booktitle'):e['venue'],'doi':e['doi']})
 for k,v in [('volume',e.get('volume')),('number',e.get('issue')),('pages',e.get('pages')),('note',e.get('publication_timing_note'))]:
  if v:f[k]=v
 bib.append('@'+typ+'{'+e['key']+str(e['year'])+',\n'+',\n'.join('  '+k+' = {'+esc(v)+'}' for k,v in f.items())+'\n}')
(P/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
n=sum(e['read_scope']=='abstract_only' for e in es)
lines=['# 海上搜救轻量语言智能体与自适应 RAG：首批参考文献','', '**范围依据：** `../海上搜救Agent_平台与研究范围决策.md`。自然语言理解、海事知识检索、观测证据引用、过期/冲突识别是论文主线；双无人机提供两路观测，双目定位为可选扩展。首批 18 篇，主线 10 篇、支撑 5 篇、扩展 3 篇。','', '**现在先处理传感器。** 同时每天读一篇主线论文，顺序建议：Adaptive-RAG → RAG → Self-RAG，再看 LMUCS/LoRA。涉及传感器的阅读先看 Kalibr 和相机标定，但不把双机改成双目作为论文必做条件。','', '## 最先读的三篇','', '1. **Adaptive-RAG（2024）**：当前方法最直接的基线，写清其按复杂度路由已有何种能力。','2. **RAG（2020，所存为预印本版）**：搭建单次检索基线之前理解知识库与生成的关系。','3. **Self-RAG（2023，所存为预印本版）**：比较按需检索、证据检查与生成检查；不要把简单自检提示等同完整复现。','', '精读时只填四格：输入输出；是否检索/再检索的依据；如何评价证据和错误；需要比较的基线。读完前三篇再判断“时效/冲突路由”是否具有独立于复杂度路由的贡献。','', '## 文献矩阵','', '|顺序/层级|论文与年份|章节用途|阅读任务/可借鉴内容|不能支持的结论|阅读范围|','|---|---|---|---|---|---|']
for e in es:
 scope='摘要' if e['read_scope']=='abstract_only' else ('书目＋README，未读论文正文' if e['key']=='lmucs' else '仅书目')
 lines.append('|'+ '|'.join([str(e['reading_order'])+'/'+e['research_role'],'['+e['title']+']('+e['url']+') ('+str(e['year'])+')',e['chapter_use'],e['reading_task'],e['cannot_support'],scope])+'|')
lines+=['','## 可导入与可追溯文件','', '- `references.bib`：18 条 BibTeX。预印本按本轮已核验版本引用，不用记忆补齐会议版本。','- `verified_metadata.json`：完整作者、年份、DOI/arXiv、摘要、阅读范围与来源文件。','- `sources/`：实际网页/API 响应与成功下载的开放 PDF。','- `scope_deferred_metadata.json`：因最新研究主线而移出的 VINS、ORB-SLAM3、YOLOv7、原始更新频率、LTE 通信资料，未计入 18 篇。','- `SEARCH_STRATEGY.md` 与 `*_log.json`：检索边界、请求记录和已纠正误匹配。','', '## LMUCS 与日期说明','', 'LMUCS 已确认期刊书目记录：Wang 等，*Aerospace Science and Technology*，DOI **10.1016/j.ast.2026.112949**。Crossref 与 Semantic Scholar 标题匹配，DOI 可解析；期次标为 **2026-10**，晚于本次检索日，DOI 于 2026-06-22 登记。精确在线发表日未确认。不能把未来期次写作已经读过的正式全文。','', '软件 README 中 YOLOv11n、MiDaS、轻量模型和指令评测属于项目自述；论文正文未读。另一个 SSRN DOI 10.2139/ssrn.5397714 的题名包含 material deliver，未合并或另计篇数。','', 'AirSim 采用 2018 论文集年（2017 在线）；HOTA 采用 2021 卷期年（2020 在线）。RAG、Self-RAG、LoRA、ReAct、SayCan、Qwen2.5、LatentRAG 目前按已取得的 arXiv 版本记录；不要由此推断它们全部未在会议发表。','', '## 证据范围与未完成项','',f'检索日期 2026-09-15。{n} 篇读摘要、{len(es)-n} 篇仅核对书目（LMUCS 另读 README）。**没有全文精读、没有 PDF 页码引用、没有人工已读认证。** PDF 下载不代表读过；下一次全文阅读前应做 read-integrity preflight。','', '这是题名种子检索与近邻补充，不是系统综述。未完成撤稿、COI、作者资历和全文方法质量审查；文献存在不等于其结论可直接应用。仍缺海事权威知识来源、过期/冲突证据识别的最新专项论文，后续需围绕确定的数据标签检索。不能依据本清单宣称创新性已成立。','', 'AI 使用披露：此清单由 AI 辅助获取来源、比对书目与整理阅读任务；各项“可借鉴内容”是论文准备建议，引用具体公式/实验结果前须精读对应版本。']
pdf_lines=['## 已下载的六份公开 PDF','','以下文件仅下载留存，尚未全文精读。','']
for label,filename in [('Adaptive-RAG','adaptive_paper.pdf'),('RAG','rag_paper.pdf'),('LoRA','lora_paper.pdf'),('AirSim','airsim_paper.pdf'),('SeaDronesSee','seadrones_paper.pdf'),('RAFT-Stereo（扩展）','raftstereo_paper.pdf')]:
 pdf_lines.append('- ['+label+'](/D:/forAirsim/docs/references-20260915/sources/'+filename+')')
pdf_lines.append('')
idx=lines.index('## LMUCS 与日期说明');lines[idx:idx]=pdf_lines
(P/'阅读指南与文献矩阵.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Final entries',len(es),'abstract_only',n,'roles',collections.Counter(e['research_role'] for e in es))
