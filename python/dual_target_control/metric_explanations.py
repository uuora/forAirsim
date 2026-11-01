"""Chinese units, definitions and decision rules for recorded control data."""
import csv,html,json
from pathlib import Path

METHOD_SUMMARY='''<h3>六组共用固定跟踪控制器，区别在参考轨迹搜索</h3>
<p>普通Bézier/Hermite：每段满足起终位置、速度、加速度的五次基线，形状扰动为0，时长按距离规则给定；两种表示可对应同一物理曲线，换基不是创新。</p>
<p>基础ABOM：调用冻结上游网络的本地适配器，联合搜索每段(dx,dy,dz,T)，N段为4N维；网络在搜索中更新。它不改变底层反馈律，也不等于完整复现原论文。</p>
<p>扰动改进候选：每代20个候选，一半由网络生成，一半由多个精英附近的正负配对扰动生成；归一化域[-1,1]，初始尺度0.15，范围0.025～0.5，按成功率调整，越界反射。属于参数搜索探索机制，不是外力抗扰控制。</p>
<p>范围：每段dx/dz为±0.8m、dy为±1.4m，基础时长3～7s，执行倍率统一计入模型搜索。基础/改进均80次整路线模型评价+1次独立选中验证；普通参考只验证1次。AirSim搜索调用为0。</p>
<h3>先看成功，再比较质量和资源</h3>
<p>共同门槛：同步位置峰值≤0.50m、终端位置≤0.25m、终端速度≤0.50m/s；全部航点位置误差≤0.50m。停稳组中间点停留结束速度≤0.30m/s；连续组按计划到点时速度≥0.50m/s；末点统一停稳。正常完成且全部通过才算任务成功。RMSE越小越好，目前无单独通过阈值。</p>
<p>可行后比较Q=0.7T/(T+5N)+0.1J/(J+30N)+0.2RMSE/(RMSE+0.25)，越低越好；J是参考轨迹jerk平方积分。规划墙钟（含进程启动、初始化、训练、验证）、飞行执行时间、模型/AirSim调用次数分别报告。速度指令限幅不代表电机饱和。</p>
<p>当前负结果：旧协议14停稳组未优于共同初始参考；连续组扰动候选时间较短，但误差和Q更差。尚无整体优越性或计算资源节省结论。下一步应在同容量消融、固定墙钟及多种子留出场景中检验筛选/停滞触发机制；过点速度和减速位置尚未参与搜索。</p>'''

METRICS=[
 ('model_search_validation_wall_s','本次模型搜索及验证内核耗时','秒','协议15规划子进程内部记录的搜索和选中方案验证；不含进程启动、导入及其他准备开销。','仅作开销分解，主要规划指标使用planning_wall_s完整墙钟。'),
 ('planning_wall_s','本次完整模型规划墙钟','秒','协议15到达准备点后，独立进程启动、导入、初始化、搜索/网络更新及选中方案验证的墙钟；含悬停轮询等待。','与准备位置收敛时间、参考执行时长分别报告；不是AirSim黑箱搜索。'),
 ('task_success','任务成功','布尔','执行正常完成，而且控制误差与全部过点判据通过。','用户中止或运行异常即不算成功；与单独的误差门槛通过不同。'),
 ('reference_duration_s','参考执行时长T','秒','参考曲线从开始到结束的仿真时间；停稳组包括预定中间停留。','满足控制门槛后越短越好；不是规划时间。'),
 ('position_rmse_m','同步位置RMSE','米','同一参考时刻的实际位置与参考位置之差，平方按时间平均后开方。','越小越好；目前没有单独RMSE通过阈值。'),
 ('max_position_error_m','峰值位置误差','米','参考时间区间内，采样同步位置误差的最大值。','≤0.50米。'),
 ('terminal_position_error_m','终端位置误差','米','在参考结束时刻T插值获得的实际位置与终点之差。','≤0.25米；不是降落后的位置。'),
 ('terminal_velocity_error_mps','终端速度误差','米/秒','参考结束时刻实际速度与要求的终端速度之差；最后一点要求静止。','≤0.50米/秒。'),
 ('speed_clipped_time_fraction','速度指令限幅比例','0至1','原始速度指令超过上限的时间占参考T的比例。','乘100得到百分比。不是电机饱和；不是越低越能证明算法更好。'),
 ('command_slew_clipped_time_fraction','指令变化率限幅比例','0至1','速度指令变化过快、被变化率约束限制的时间比例。','不是实际加速度超过上限的比例。'),
 ('quality_term','综合质量Q','无量纲','时间、参考jerk能量与跟踪误差的预先固定加权分数。','控制通过后，同规则同场景内越低越好；两组规则不直接混排。'),
 ('historical_planning_wall_s','所选方法历史模型规划耗时','秒','生成冻结方案时，该方法的模型搜索和完整验证耗时。','不是本次AirSim飞行时在线搜索；重复飞行不等于重新训练。'),
 ('full_model_calls','完整模型调用数','次','该方法完成的整条路线代理仿真评价数，含选中方案的验证。','不是AirSim搜索调用数。每次当前飞行只执行最终冻结参考。'),
 ('neural_generation_s','网络候选生成耗时','秒','历史模型搜索中调用网络生成候选的计时。','已包含在历史搜索耗时内，不能重复相加。'),
 ('neural_update_s','网络更新耗时','秒','历史模型搜索中网络参数更新的计时。','不是飞行时底层控制反馈；已包含在搜索耗时内。'),
 ('airsim_search_calls','AirSim搜索评价数','次','优化过程中用AirSim评估候选的次数。','当前协议为0；简化模型评价不能记成AirSim评价。'),
 ('airsim_reference_execution_calls','冻结参考AirSim执行数','次','本次运行是否进入冻结参考的AirSim执行阶段。','不包含起飞、准备和降落；不能表示控制RPC请求数。'),
 ('tracking_host_wall_s','实测跟踪墙钟时间','秒','计算机上从第一条跟踪状态到终点收敛的时间，包括RPC等开销。','不含起飞和降落；与参考T、规划耗时分开看。'),
 ('waypoint_position_error_m','计划过点时刻的位置误差','米','按该航点预定核验时刻对实际状态插值得到的位置偏差。','≤0.50米；逐点停稳组在预定停留结束时核验。'),
 ('waypoint_actual_speed_mps','计划过点时刻的实际速度','米/秒','核验时刻实际速度向量的长度。','中间点：停稳组≤0.30，连续组≥0.50；最后一点统一≤0.50。'),
]

FIELDS=[('t_s','参考经过时间','秒','按AirSim状态时间戳推进；从本条参考起点计时。'),
 ('px/py/pz','实际位置','米','x、y、向上高度；不是NED向下Z。'),('vx/vy/vz','实际速度','米/秒','状态估计中的平移速度。'),
 ('qx/qy/qz','参考位置','米','同一时刻要求到达的位置；q不表示实际状态。'),('qvx/qvy/qvz','参考速度','米/秒','参考位置对物理时间的一阶导数。'),
 ('ax/ay/az','估计实际加速度','米/秒²','AirSim估计状态；不等于速度指令变化率，也不是电机信号。'),
 ('cmd_x/cmd_y/cmd_z','发送的速度指令','米/秒','经速度和变化率限制后发送给控制API；该行状态之后发出。'),
 ('yaw_rad','实测机头偏航','弧度','乘180/π转换为角度；是状态估计，不是指令。'),
 ('yaw_command_deg','机头偏航指令','度','发送的绝对偏航目标；与yaw_rad单位不同。'),
 ('speed_clipped/slew_clipped','单次限幅标记','布尔','True表示本次原始指令触发对应限制。'),('paused','用户暂停标记','布尔','暂停过的记录保留，但不作为不中断轨迹的有效排名样本。')]


def glossary_section():
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [label,key,unit,meaning,rule])+'</tr>' for key,label,unit,meaning,rule in METRICS)
    fields=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in FIELDS)
    return '<section id="definitions"><h2>这些数据是什么意思？</h2><p><b>先看执行状态，再看控制判据，最后比较质量和耗时。</b>“完成”只表示执行流程完成；控制通过还要求全部误差与过点规则同时满足。没有有效分数时用空值，不能当作0误差。</p><div class="scroll"><table><tr><th>中文指标</th><th>CSV字段</th><th>单位</th><th>怎样计算/表示什么</th><th>怎样判断</th></tr>'+rows+'</table></div><h3>原始states.csv字段</h3><div class="scroll"><table><tr><th>字段</th><th>含义</th><th>单位</th><th>说明</th></tr>'+fields+'</table></div><p>逐点停稳与连续过点是两个不同任务。只有相同航点、规则、参数和预算的记录才进入同一对比；Bézier/Hermite可以表示相同物理曲线，重叠不代表显示错误。</p></section>'


def write_glossary(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    text='<html lang="zh-CN"><meta charset="utf-8"><title>控制实验数据说明</title><style>body{font:15px system-ui;margin:28px;color:#203348}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dae3ea;padding:10px;text-align:left}.scroll{overflow:auto}</style>'+METHOD_SUMMARY+glossary_section()+'</html>'
    (folder/'index.html').write_text(text,encoding='utf-8')
    with (folder/'数据字段说明.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['字段','中文名称','单位','定义','判断方式']);w.writerows(METRICS)
    (folder/'definitions.json').write_text(json.dumps(dict(metrics=METRICS,states=FIELDS),ensure_ascii=False,indent=2),encoding='utf-8')
    return (folder/'index.html').resolve()
