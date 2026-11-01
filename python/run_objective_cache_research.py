"""Paired exact-cache baseline for deterministic offline curve objectives.

The experiment preserves candidate/elite/model streams and counts underlying
oracle calls separately from returned scores for proposals. Memoization is a
standard engineering baseline, not a novel optimizer or a flight experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from trajectory_research.blackbox import run_blackbox_search
from trajectory_research.objective_cache import DeterministicObjectiveCache
from trajectory_research.parameterizations import BoundaryStates, default_parameters, parameter_bounds
from trajectory_research.perturbation_task import declaration, evaluate_parameters, get_layout
from trajectory_research.screened_search import run_screened_search
from trajectory_research.search import UPSTREAM_COMMIT, UPSTREAM_NETWORK_BLOB, run_search


ROOT=Path(__file__).resolve().parents[1]
CONFIGS=("abom_p095","abom_p010","gaussian","coordinate")
DROPOUT={"abom_p095":.95,"abom_p010":.1}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def validate_args(args):
    if args.budget<16 or args.budget%16:
        raise ValueError("budget must be a positive multiple of population=16")
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or any(seed<0 for seed in args.seeds):
        raise ValueError("distinct nonnegative seeds are required")
    if not args.configs or len(set(args.configs))!=len(args.configs) or any(c not in CONFIGS for c in args.configs):
        raise ValueError("distinct declared cache-study configurations are required")
    if not args.layouts or len(set(args.layouts))!=len(args.layouts):
        raise ValueError("distinct layouts are required")
    for layout in args.layouts: get_layout(layout)
    if not args.families or len(set(args.families))!=len(args.families):
        raise ValueError("distinct curve families are required")
    for family in args.families: parameter_bounds(family)
    if args.max_entries is not None and args.max_entries<1:
        raise ValueError("max_entries must be positive or omitted")


def run_optimizer(objective,lower,upper,config,args,seed,anchor):
    options=dict(seed=seed,population=16,budget=args.budget,anchor=anchor,initialization="uniform")
    if config in DROPOUT:
        return run_screened_search(objective,lower,upper,method="abom",upstream=args.upstream,
            lower_bound=None,bounded_objective=None,dropout=DROPOUT[config],**options)
    if config=="gaussian":
        return run_search(objective,lower,upper,method="gaussian",**options)
    return run_blackbox_search(objective,lower,upper,method="coordinate",**options)


def assert_same_stream(plain,memoized,ledger,budget):
    for field in ("initial_population_normalized","best_parameters","best_score","history"):
        assert plain[field]==memoized[field],f"changed search state: {field}"
    assert len(plain["evaluations"])==len(memoized["evaluations"])==len(ledger)==budget
    measured={}
    for a,b,entry in zip(plain["evaluations"],memoized["evaluations"],ledger):
        assert a["parameters"]==b["parameters"]==entry["parameters"]
        assert a["score"]==b["score"]==entry["score"]
        assert a["feasible"]==b["feasible"]==entry["feasible"]
        source=entry["source_evaluation_id"]
        if entry["status"]=="FULL_EVALUATION":
            assert source not in measured
            measured[source]=entry
        else:
            assert entry["status"]=="CACHE_HIT" and source in measured
            old=measured[source]
            assert old["key_digest"]==entry["key_digest"]
            assert old["parameters"]==entry["parameters"]
            assert old["score"]==entry["score"] and old["feasible"]==entry["feasible"]
            assert entry["oracle_wall_s"]==0. and entry["result_reused"]
    assert len(measured)==sum(e["status"]=="FULL_EVALUATION" for e in ledger)


def summaries(rows,args):
    result=[]
    for family in args.families:
        for layout in args.layouts:
            for config in args.configs:
                group=[r for r in rows if (r["family"],r["layout"],r["config"])==(family,layout,config)]
                assert len(group)==len(args.seeds)
                ratios=np.array([r["memoized_wall_s"]/r["plain_wall_s"] for r in group])
                rng=np.random.default_rng(20261029)
                boot=np.median(ratios[rng.integers(0,len(group),(5000,len(group)))],axis=1)
                result.append(dict(family=family,layout=layout,config=config,n=len(group),
                    median_wall_ratio=float(np.median(ratios)),
                    exploratory_paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    oracle_calls_plain=sum(r["plain_oracle_calls"] for r in group),
                    oracle_calls_memoized=sum(r["memoized_oracle_calls"] for r in group),
                    reused_measurement_fraction=sum(r["cache_hits"] for r in group)/sum(r["proposal_budget"] for r in group),
                    median_peak_cache_entries=float(np.median([r["peak_cache_entries"] for r in group])),
                    feasible_selected=sum(r["feasible"] for r in group),
                    all_streams_identical=all(r["stream_identical"] for r in group)))
    return result


def run(args):
    validate_args(args)
    source_names=("objective_cache.py","search.py","screened_search.py","blackbox.py",
        "perturbation_task.py","parameterizations.py","curves.py","curve_task.py","constraints.py","directional.py")
    sources=[Path(__file__)]+[ROOT/"python/trajectory_research"/name for name in source_names]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    neural=any(c in DROPOUT for c in args.configs)
    upstream=None
    if neural:
        data=(Path(args.upstream).resolve()/"abom_net.py").read_bytes()
        blob=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if blob!=UPSTREAM_NETWORK_BLOB: raise ValueError("pinned upstream network Git blob differs")
        upstream=dict(commit=UPSTREAM_COMMIT,network_git_blob=blob,network_sha256=hashlib.sha256(data).hexdigest())
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for source in sources: shutil.copy2(source,snapshot/source.name)
    plan=[]
    for family in args.families:
        for layout in args.layouts:
            for seed in args.seeds:
                shift=len(plan)%len(args.configs)
                order=list(args.configs[shift:]+args.configs[:shift])
                plan.append(dict(family=family,layout=layout,seed=seed,config_order=order,
                    arm_order=["plain","memoized"] if seed%2 else ["memoized","plain"]))
    contexts=[]
    for family in args.families:
        for layout in args.layouts:
            description=dict(objective="trajectory_research.perturbation_task.evaluate_parameters",
                objective_version="endpoint-preserving-geometry-v1",family=family,layout=asdict(get_layout(layout)),
                robust=True,representation="hermite",default_endpoint_states=asdict(BoundaryStates()),
                task_declaration=declaration(),source_sha256=hashes,
                python=platform.python_version(),numpy=np.__version__)
            encoded=json.dumps(description,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
            contexts.append(dict(family=family,layout=layout,context_id="sha256:"+hashlib.sha256(encoded).hexdigest(),
                                 description=description))
    manifest=dict(protocol="paired-deterministic-objective-cache-v1",args=vars(args),source_sha256=hashes,
        upstream=upstream,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        execution_plan=plan,expected_pairs=len(plan)*len(args.configs),population=16,
        configurations={c:dict(method="abom" if c in DROPOUT else c,dropout=DROPOUT.get(c)) for c in args.configs},
        cache_contexts=contexts,cache_policy="exact float64 bits, no quantization; new instance per memoized arm; explicit deterministic contract",
        screening_enabled=False,
        budget="fixed proposals; actual underlying oracle calls separately counted; optimizer fe/full_evaluations fields are nominal proposal records in the cached arm",
        timing="warm process; cache construction, lookup, deep copies, provenance collection and all optimizer/oracle work included; ledger export, serialization and post-selection recheck excluded",
        scope="offline deterministic point-reference geometry; no controller, stochastic simulator, AirSim or flight experiment",
        novelty="standard memoization baseline; no novelty or global superiority claim")
    write_json(output/"manifest.json",manifest)
    if neural:
        warm=time.perf_counter()
        import torch
        torch.set_num_threads(1)
        torch.zeros(1).sum()
        run_screened_search(lambda x:dict(score=float(x@x),feasible=True),np.full(4,-1.),np.full(4,1.),
            method="abom",budget=32,upstream=args.upstream,dropout=.95)
        manifest["excluded_import_optimizer_warmup_s"]=time.perf_counter()-warm
        manifest["synthetic_warmup_oracle_calls"]=32
        manifest["torch_version"]=torch.__version__
        write_json(output/"manifest.json",manifest)
    rows=[]
    anchors=[]
    warmed=set()
    for block in plan:
        family,layout,seed=block["family"],block["layout"],block["seed"]
        lower,upper=parameter_bounds(family)
        anchor=default_parameters(family)
        context=next(c for c in contexts if (c["family"],c["layout"])==(family,layout))
        objective=lambda x:evaluate_parameters(x,family=family,layout=layout,representation="hermite",robust=True)
        if (family,layout) not in warmed:
            anchors.append(dict(family=family,layout=layout,result=objective(anchor)))
            write_json(output/"anchor_checks.json",anchors)
            warmed.add((family,layout))
        common_initial=None
        for config in block["config_order"]:
            results={}
            for arm in block["arm_order"]:
                calls=0
                def counted(x):
                    nonlocal calls
                    calls+=1
                    return objective(x)
                start,cpu=time.perf_counter(),time.process_time()
                memoizer=None
                if arm=="memoized":
                    memoizer=DeterministicObjectiveCache(counted,context["context_id"],
                        deterministic=True,max_entries=args.max_entries)
                search=run_optimizer(counted if memoizer is None else memoizer,lower,upper,config,args,seed,anchor)
                wall_s,cpu_s=time.perf_counter()-start,time.process_time()-cpu
                assert len(search["evaluations"])==args.budget
                assert search["best_score"]==min(e["score"] for e in search["evaluations"])
                if common_initial is None: common_initial=search["initial_population_normalized"]
                else: assert search["initial_population_normalized"]==common_initial
                cache_stats=None if memoizer is None else memoizer.stats
                ledger=[] if memoizer is None else memoizer.ledger
                assert calls==(args.budget if memoizer is None else cache_stats["oracle_calls"])
                if memoizer is not None:
                    assert cache_stats["proposals"]==args.budget
                    assert calls+cache_stats["hits"]==args.budget
                    assert cache_stats["failed_evaluations"]==0
                results[arm]=dict(search=search,wall_s=wall_s,cpu_s=cpu_s,actual_oracle_calls=calls,
                    cache_stats=cache_stats,cache_ledger=ledger,
                    returned_score_accounting="one returned score per proposal; a cache hit reuses a prior full measurement")
            plain,memoized=results["plain"],results["memoized"]
            assert_same_stream(plain["search"],memoized["search"],memoized["cache_ledger"],args.budget)
            checked=objective(plain["search"]["best_parameters"])
            assert checked["score"]==plain["search"]["best_score"]
            stats=memoized["cache_stats"]
            row=dict(family=family,layout=layout,config=config,seed=seed,proposal_budget=args.budget,
                dropout=DROPOUT.get(config),best_score=checked["score"],feasible=checked["feasible"],
                plain_oracle_calls=plain["actual_oracle_calls"],memoized_oracle_calls=memoized["actual_oracle_calls"],
                cache_hits=stats["hits"],cache_entries=stats["cache_entries"],peak_cache_entries=stats["peak_cache_entries"],
                cache_evictions=stats["evictions"],plain_wall_s=plain["wall_s"],memoized_wall_s=memoized["wall_s"],
                plain_cpu_s=plain["cpu_s"],memoized_cpu_s=memoized["cpu_s"],
                stream_identical=True,post_selection_oracle_calls=1,cache_context_id=context["context_id"])
            stem=f"{family}_{layout}_{config}_{seed}"
            write_json(output/f"{stem}.json",dict(summary=row,arm_order=block["arm_order"],
                selected_recheck=checked,**results))
            rows.append(row)
            write_json(output/"rows.json",rows)
            print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
    assert len(rows)==manifest["expected_pairs"]
    for source in sources:
        assert hashlib.sha256(source.read_bytes()).hexdigest()==hashes[source.relative_to(ROOT).as_posix()], \
            "source changed during experiment; preserve as unbound diagnostics"
    write_json(output/"summary.json",summaries(rows,args))
    write_json(output/"completion.json",dict(status="COMPLETE",pairs=len(rows),search_runs=len(rows)*2,
        proposals_per_arm=sum(r["proposal_budget"] for r in rows),
        actual_oracle_calls_plain=sum(r["plain_oracle_calls"] for r in rows),
        actual_oracle_calls_memoized=sum(r["memoized_oracle_calls"] for r in rows),
        cache_hits=sum(r["cache_hits"] for r in rows),post_selection_oracle_calls=len(rows),
        anchor_warmup_calls=len(anchors),all_streams_identical=True,source_bytes_unchanged=True,
        memory_scope="entry counts only; resident memory and energy savings were not measured"))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--budget",type=int,default=256)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(701,711)))
    parser.add_argument("--families",nargs="+",default=["single_bump"])
    parser.add_argument("--layouts",nargs="+",default=["central","early_offset","two_obstacles"])
    parser.add_argument("--configs",nargs="+",default=list(CONFIGS))
    parser.add_argument("--max-entries",type=int,default=None)
    parser.add_argument("--upstream",default=str(ROOT/"tmp/abom_deploy_probe/upstream"))
    run(parser.parse_args())
