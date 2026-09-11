"""Publication-readable scientific figures and data-derived interpretation."""
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .visualization import configure_matplotlib
from .causal_scenarios import CausalScenarioFactory


def _save(fig,path,caption):
    lines=textwrap.wrap(caption,width=64)
    fig.text(.08,.025,"\n".join(lines),ha="left",va="bottom",fontsize=10,color="#334155")
    fig.tight_layout(rect=(.02,.07+.022*len(lines),.98,.97))
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)


def create_figures(output,data,q1,info,modes,config):
    configure_matplotlib()
    figures=output/"figures";figures.mkdir(exist_ok=True)
    captions=[]
    def save(fig,filename,caption):
        _save(fig,figures/filename,caption)
        captions.append((filename,caption))
    hour=np.arange(144)/6
    blue,green,orange,red="#28649A","#2E8B70","#D58A32","#C44E52"
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for col,label,color in (("load_energy_kwh","负载", "#718096"),("pv_forecast_energy_kwh","光伏",green),("purchase_kwh","计划购电",blue)):
        axes[0].plot(hour,q1[col],label=label,color=color)
    axes[0].bar(hour,q1.charge_kwh,width=.14,label="充电",color=orange,alpha=.6)
    axes[0].bar(hour,-q1.discharge_kwh,width=.14,label="放电（负向展示）",color=red,alpha=.65)
    axes[0].set(title="问题1｜典型日最优能量调度",ylabel="时段电量（kWh / 10分钟）")
    axes[1].plot(np.arange(145)/6,np.r_[6000.,q1.storage_kwh],label="储电量",color=blue)
    axes[1].axhline(1200,color=red,ls="--",label="SOC下限1200 kWh")
    axes[1].axhline(10800,color=orange,ls="--",label="SOC上限10800 kWh")
    axes[1].set(xlabel="时刻（小时）",ylabel="储电量（kWh）",xlim=(0,24),xticks=np.arange(0,25,4))
    for ax in axes:ax.legend(ncol=3,fontsize=9);ax.grid(alpha=.2)
    first=info["question1"]
    saving=100*(1-first["cost_yuan"]/first["baseline_cost_yuan"])
    save(fig,"01_typical_dispatch.png",f"图1：典型日购电成本由无储能基线的{first['baseline_cost_yuan']:,.2f}元降至{first['cost_yuan']:,.2f}元，下降{saving:.2f}%；末态回到6000 kWh。")

    day=min(78,30+len(modes["Q2"].dates));index=day-31
    factory=CausalScenarioFactory(data,config,info["solver_parameters"]["scenarios"])
    forecast=factory.get(day)
    actual=data.net_kwh[day]
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    axes[0].fill_between(hour,forecast.lower,forecast.upper,color=blue,alpha=.15,label="历史分割校准带（名义90%）")
    axes[0].plot(hour,actual,color="#374151",label="实际净负荷")
    axes[0].plot(hour,forecast.center,color=blue,label="日前组合预测")
    axes[0].set(title=f"问题2｜{data.dates[day]:%Y-%m-%d}日前预测与执行",ylabel="净负荷电量（kWh / 10分钟）")
    axes[1].plot(hour,modes["Q2"].original_plan_kwh[index],label="原始计划",color=blue)
    axes[1].plot(hour,modes["Q2"].used_plan_kwh[index],label="实际使用计划电",color=green)
    axes[1].bar(hour,modes["Q2"].emergency_kwh[index],width=.14,label="紧急购电",color=red)
    axes[1].set(xlabel="时刻（小时）",ylabel="时段电量（kWh / 10分钟）",xlim=(0,24),xticks=np.arange(0,25,4))
    for ax in axes:ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
    coverage=100*np.mean((actual>=forecast.lower)&(actual<=forecast.upper))
    mae=float(np.abs(actual-forecast.center).mean())
    save(fig,"02_forecast_execution.png",f"图2：代表日净负荷预测MAE为{mae:.2f} kWh/时段，校准带实际覆盖{coverage:.2f}%。预测带仅作诊断；购电量来自场景DRO求解，不等于预测上界。")

    summary=pd.DataFrame(info["models"]).set_index("model")
    for k,(names,title,filename) in enumerate(((('Q2','Q3'),'问题2与问题3｜固定电价成本构成','03_fixed_costs.png'),(('Q4-2','Q4-3'),'问题4｜变动电价成本构成','04_variable_costs.png')),start=3):
        fig,axes=plt.subplots(1,2,figsize=(13,7),gridspec_kw={"width_ratios":[1,1.7]})
        bottom=np.zeros(2)
        for col,label,color in (("plan_cost_yuan","原始计划费用",blue),("adjustment_cost_yuan","调整净费用",orange),("emergency_cost_yuan","紧急购电费用",red)):
            values=summary.loc[list(names),col].to_numpy()/1e4
            axes[0].bar(names,values,bottom=bottom,label=label,color=color);bottom+=values
        for x,value in enumerate(bottom):axes[0].text(x,value,f"{value:,.1f}",ha="center",va="bottom")
        axes[0].set(title=title,ylabel="累计实际费用（万元）",xlabel="策略",ylim=(0,max(bottom)*1.15))
        axes[0].legend(fontsize=9,loc="lower right")
        for name,color in zip(names,[blue,orange]):
            daily=modes[name].daily.copy();daily["month"]=pd.to_datetime(daily.date).dt.month
            monthly=daily.groupby("month").total_cost_yuan.sum()/1e4
            axes[1].plot(monthly.index,monthly.values,marker="o",label=name,color=color)
        axes[1].set(title="按月实际总费用",xlabel="月份",ylabel="月费用（万元）",xticks=monthly.index)
        axes[1].legend();axes[1].grid(alpha=.2)
        difference=100*(1-summary.loc[names[1],"total_cost_yuan"]/summary.loc[names[0],"total_cost_yuan"])
        direction="下降" if difference>=0 else "上升"
        save(fig,filename,f"图{k}：{names[1]}相比{names[0]}累计费用{direction}{abs(difference):.2f}%。两者的信息集和滚动决策机制同时不同，不能把差额全部归因于价值门控。")

    fig,axes=plt.subplots(1,2,figsize=(13,7))
    for name,color in (("Q3",blue),("Q4-3",orange)):
        counts=modes[name].gates.groupby("release_time").accepted.sum() if len(modes[name].gates) else pd.Series(dtype=float)
        position=np.arange(len(counts))+( -.18 if name=="Q3" else .18)
        axes[0].bar(position,counts.to_numpy(),width=.35,label=name,color=color)
    axes[0].set(title="日内购电调整门控",xlabel="预报发布时间",ylabel="接受调整次数",xticks=np.arange(3),xticklabels=["06:00","12:00","18:00"])
    axes[0].legend();axes[0].grid(axis="y",alpha=.2)
    no_update=[name for name in summary.index if "no-update" in name]
    if no_update:
        labels=["Q3-no-update","Q3","Q4-3-no-update","Q4-3"]
        axes[1].bar(np.arange(4),summary.loc[labels,"total_cost_yuan"]/1e4,color=[green,blue,green,orange],label="累计实际费用")
        axes[1].set(xticks=np.arange(4),xticklabels=["固定价\n不更新","固定价\n门控滚动","变动价\n不更新","变动价\n门控滚动"],ylabel="累计实际费用（万元）",xlabel="相同00:00信息，是否进行日内更新",title="信息匹配的运行基线")
        axes[1].legend()
        fixed_gain=100*(1-summary.loc['Q3','total_cost_yuan']/summary.loc['Q3-no-update','total_cost_yuan'])
        variable_gain=100*(1-summary.loc['Q4-3','total_cost_yuan']/summary.loc['Q4-3-no-update','total_cost_yuan'])
        caption=f"图5：相同00:00预报信息下，门控滚动相对不做日内更新的费用变化为固定价{-fixed_gain:+.2f}%、变动价{-variable_gain:+.2f}%。该对比同时包含日内新预报、储能重优化和购电门控的作用。"
    else:
        labels=["Q2","Q3","Q4-2","Q4-3"]
        axes[1].bar(labels,summary.loc[labels,"emergency_energy_kwh"]/1e3,color=[green,blue,green,orange],label="紧急购电")
        axes[1].set(title="紧急购电量",xlabel="策略",ylabel="紧急购电量（千kWh）");axes[1].legend()
        caption="图5：门控次数反映购电计划更新频率；右图比较实际紧急购电量。次数较多本身不等于经济性更好。"
    save(fig,"05_gate_and_baseline.png",caption)
    return captions


def write_report(output,info,captions):
    summary=pd.DataFrame(info["models"])
    lines=["# C题：Python模型求解与结果说明（修正版）","",
        "本报告对应 `src/sxjm/solve_verified_models.py`。旧版 `outputs/model_solution` 为分位数加裕度的启发式基线，不能当作DRO/CVaR精确求解结果。修正版数值以本目录为准。",
        "","## 1. 求解范围及必须说明的假设","",
        "1. Q1按题给典型日确定性净负荷求解；Q2—Q4均逐日、逐发布时间回放，只允许历史实际值及当时已发布的预报进入决策。电量采用kWh/10分钟，功率约束用5000×1/6转换为每时段833.333 kWh。",
        "2. 充、放电效率分别取0.9，往返效率为0.81；SOC范围1200—10800 kWh。若题意0.9指往返效率，则应改成两侧sqrt(0.9)后重算，不能混用。",
        "3. 1月用于历史初始化，电池保持6000 kWh，购电计划采用前一天正净负荷，缺口紧急购买；1月1日无历史时q=0。记录见january_initialization.csv。2月1日继承1月31日实际末态，未凭空重置SOC；此初始化不是1月最优调度。",
        "4. 主口径不售电，未使用的计划电仍需付款；调减退款系数rho由运行参数给出。调整费用按各时段最终计划相对原始计划计算，而非把反复调整作为独立交易累加。若题目要求逐次结算，必须换结算模型。",
        f"5. 本次第四问电价信息采用 **{info['solver_parameters']['price_information']}**：known表示00:00已知当日各时段电价；unknown表示仅由历史预测价格。其余可调参数见solution_summary.json。",
        "", "## 2. 数据输入 → 参数初始化 → 模型调用 → 结果输出", "",
        "|步骤|对应Python位置|注意事项|","|---|---|---|",
        "|数据输入|solve_all_models.load_inputs；causal_scenarios.CausalScenarioFactory|读取未标准化的物理量CSV；检查365×144及各发布时间的未来可用段；不使用processed_full缩放量。|",
        "|参数初始化|model_config.ModelConfig；主程序命令行参数|效率在(0,1]；CVaR置信水平在(0,1)；rho、lambda在[0,1]；epsilon当前限制在[0,1]。|",
        "|确定性模型|optimization.solve_energy_plan|Q1稀疏能量流LP；末态6000；检查功率、SOC、互斥残差。|",
        "|随机模型|stochastic.solve_stochastic_plan|有限支持运输距离DRO+CVaR；先LP，若充放同时为正则启用二元变量重解MILP。|",
        "|滚动调用|solve_verified_models.run_mode|06/12/18点只改未执行时段；候选模型直接包含调增、调减费用；门控不接受时仍允许储能在原购电计划下重优化。|",
        "|结果输出|write_workbook；verify_mode；verify_workbook；verified_outputs|输出全时段CSV、全日期表格和图；独立重算收支、连续SOC、费用；原始数据与模板SHA-256前后校验一致。|",
        "", "### 关键参数", "",
        f"- CVaR置信水平 alpha={info['parameters']['risk_alpha']}；风险权重 lambda={info['solver_parameters']['risk_weight']}；Wasserstein半径 epsilon={info['solver_parameters']['radius']}；支持场景数上限={info['solver_parameters']['scenarios']}。",
        "- epsilon是对标准化整日轨迹距离的限制，不是预测误差百分比。lambda=0得到纯最坏期望；epsilon=0得到所选支持的经验分布风险模型。",
        f"- 紧急电价倍率5、调增倍率1.5、调减处罚倍率0.5、退款系数rho={info['parameters']['down_refund_ratio']}。门控经验自助法使用300次重采样和固定种子，阈值为保留方案情景平均总费用的0.1%。",
        "- 参数应仅用考核日前的训练/验证窗口选择；本文参数是透明的预设值，不声称最优调参。名义90%历史校准带只用于诊断，不作为硬保供承诺。",
        "", "## 3. 公式与代码对应", "",
        "### 3.1 物理层", "",
        "对每时段，q为付费计划，x为实际使用计划电，h为紧急购电，c/d为充/放电，s为弃光：", "",
        r"$$x_t+h_t+d_t-c_t-s_t=N_t,\qquad 0\le x_t\le q_t.$$", "",
        r"$$E_t=E_{t-1}+0.9c_t-d_t/0.9,\quad 1200\le E_t\le10800.$$", "",
        r"$$0\le c_t\le B y_t,\quad0\le d_t\le B(1-y_t),\quad B=5000/6,\quad y_t\in\{0,1\}.$$", "",
        "优化使用公共c、d及E路径。情景紧急电量满足h_st≥N_st+c_t-d_t-q_t；同时限制d_t−c_t≤max(min_s N_st,0)，禁止用弃电吸收电池放电。这一额外保守限制让有限场景共同动作可行。执行时只根据当前实际净负荷和SOC裁剪储能动作，偏差量逐日记录，不读取未来实际值。",
        "", "### 3.2 成本及DRO/CVaR", "",
        "原始计划费用p·q0始终计入；滚动阶段令u=max(q−q0,0)、v=max(q0−q,0)：", "",
        r"$$C_s=\sum_t [p_{st}q_t^0+1.5p_{st}u_t+(0.5-\rho)p_{st}v_t+5p_{st}h_{st}].$$", "",
        "日前阶段用p·q替代前三项。代码以q−u+v=q0及u,v≥0线性化；两侧费用斜率使同时调增调减没有收益。终端SOC计划目标统一6000 kWh，实际SOC不重置，下一次求解从实际状态开始。",
        "", "支持情景概率为pi_j。运输矩阵gamma将经验质量搬到同一有限支持，满足sum_s gamma_js=pi_j、sum_js D_js gamma_js≤epsilon。目标为：", "",
        r"$$\min_a\ \sup_{P\in\mathcal P}\{\mathbb E_P[C(a)]+\lambda\operatorname{CVaR}_{\alpha,P}(C(a))\}.$$", "",
        "CVaR引入zeta、xi_s≥C_s−zeta、xi_s≥0；运输线性规划对偶引入theta≥0和自由beta_j：", "",
        r"$$\min\ \lambda\zeta+\epsilon\theta+\sum_j\pi_j\beta_j,$$", "",
        r"$$\beta_j\ge C_s+\lambda\xi_s/(1-\alpha)-\theta D_{js},\quad\forall j,s.$$", "",
        "有限概率单纯形上的凸紧性允许交换CVaR阈值最小化与最坏分布最大化；随后由运输LP对偶得到上述有限规划。另加1e−6元/kWh充放吞吐惩罚用于数值破同优，其值不计入实际电费。共同动作、有限支持及校准策略均是明确近似，不能据此声称求解了无限支持、完整自适应多阶段模型的全局最优解。",
        "", "### 3.3 预测、Copula与门控", "",
        "日前预测由最近7天与同星期历史中位数加权组成；权重用前14天已发生的预报误差计算。日内负载按已观察到的当日前缀进行0.8—1.2倍修正，配合当时发布的光伏预报。历史残差均针对该历史日自身当时可得的预测计算，不拿当前预测去回拟过去。",
        "", "最近56天残差按整日向量保留。未知电价模式把同一历史日的净负荷残差和电价残差成对取样，即经验联合分布/经验Copula的离散实现；不独立打乱价格或时段。known模式电价退化为已知路径，此时不额外拟合虚假的随机价格相关性。支持缩减采用确定性最远点代表轨迹及最近簇频数权重；它近似完整经验分布，不声称精确保留所有秩相关。",
        "", "价值门控在同一场景集上比较固定购电与可调整购电两种优化。接受条件为：经验自助法节省下分位数超过阈值，或90%尾部紧急量超过50 kWh且候选降低至少5%。这里的下分位数是训练场景诊断，不是经过样本外认证的置信下界；时间依赖、聚类缩减及同样本优化均可能造成偏差。",
        "", "## 4. 实际运行结果", "",
        f"计算期：{'部分测试，不能当全年结果' if info['partial'] else '2025年2月1日至12月31日，共334天'}。单位为人民币元、kWh；实际费用不含CVaR风险溢价与数值破同优惩罚。", "",
        f"Q1无储能基线：{info['question1']['baseline_cost_yuan']:,.2f}元；确定性优化：{info['question1']['cost_yuan']:,.2f}元。", "",
        "|策略|原始计划费/元|调整净费/元|紧急购电费/元|实际总费/元|紧急购电量/kWh|接受门控次数|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"|{row.model}|{row.plan_cost_yuan:,.2f}|{row.adjustment_cost_yuan:,.2f}|{row.emergency_cost_yuan:,.2f}|{row.total_cost_yuan:,.2f}|{row.emergency_energy_kwh:,.2f}|{row.accepted_gate_count}|")
    lines += ["","no-update为相同00:00信息但不做日内预报/储能/购电更新的基线；不应把与Q2的费用差完全归因于门控。它也不是无门控的持续调整基线。",
        "","### 可视化与关键结论",""]
    for filename,caption in captions:lines += [f"![{caption}](figures/{filename})","",caption,""]
    lines += ["## 5. 可行性与输出校验", "",
        "|策略|能量平衡最大残差/kWh|SOC递推最大残差/kWh|日间SOC跳变/kWh|同时充放/kWh|期末SOC/kWh|",
        "|---|---:|---:|---:|---:|---:|"]
    for name,check in info["physical_checks"].items():
        lines.append(f"|{name}|{check['max_balance_residual_kwh']:.3g}|{check['max_soc_residual_kwh']:.3g}|{check['max_day_boundary_residual_kwh']:.3g}|{check['max_simultaneous_kwh']:.3g}|{check['final_soc_kwh']:.3f}|")
    lines += ["", "全部时段检查：功率/SOC上下限、实际计划电≤付费计划、弃光≤实际光伏盈余、非负性、全量费用重算、原文件哈希一致。solver_audit.csv给出优化残差及MILP状态，forecast_coverage.csv给出逐日预测覆盖率。",
        "", "原模板部分时段标签偏移10分钟：输出副本改为00:00—24:00并提供template_time_mapping.csv。充放电展开为334×6行，紧急电展开为所有连续事件，不把‘其余时段合计’塞进示例行。该格式是完整计算交付版，正式提交前需确认赛方是否允许改标签/扩展示例行；不作已通过官方格式审核的承诺。",
        "", "计划电量合计列为SUM公式，openpyxl不计算公式缓存，首次Excel打开会重算；核验脚本直接核对公式引用及其144个源数值，而非把缺失缓存当零。",
        "", "## 6. 一键复现", "", "```powershell", "# 工作目录 D:\\mathmodel\\SXJM；依赖见pyproject.toml及uv.lock", "uv sync", "uv run python -m sxjm.solve_verified_models --output outputs/model_solution_v2 --ablations", "", "# 先做2天小测试（写入独立目录）", "uv run python -m sxjm.solve_verified_models --days 2 --output outputs/smoke_verified", "", "# 电价未知或调减退款的口径敏感性，必须写入另一个目录", "uv run python -m sxjm.solve_verified_models --price-information unknown --output outputs/unknown_price", "uv run python -m sxjm.solve_verified_models --refund 1 --output outputs/refund_one", "```", "",
        "源代码模块：solve_verified_models.py（主流程/导出/校验）、stochastic.py（核心规划/因果执行）、causal_scenarios.py（预测/校准/联合场景）、verified_outputs.py（图/报告），另复用model_config.py和optimization.py。",
        "", "## 7. 解释边界与参考", "",
        "本次是附件上的历史回放，不是未接触的严格独立测试集；初版开发曾查看个别年末样本。场景数、半径、风险权重及终端策略尚不能认为是经独立验证的最优选择。有限场景外极端事件仍可能造成紧急购电。运行结果可复现，不代表未来收益保证。",
        "", "求解接口参见[SciPy linprog官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)；Wasserstein可处理化方法背景参见[Esfahani与Kuhn原论文](https://arxiv.org/abs/1505.05116)。本文有限支持对偶由上节给出，不借用原论文的有限样本保证来替本数据背书。", ""]
    (output/"模型求解与结果说明.md").write_text("\n".join(lines),encoding="utf-8")
