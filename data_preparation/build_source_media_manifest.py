#!/usr/bin/env python3
"""Create upstream media identifiers without exposing local media paths."""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
from typing import Any
SOURCE_NAMES={"AI2THOR":"AI2-THOR","PROC":"ProcTHOR","REP":"ReplicaCAD","arkit":"ARKitScenes","ScanNetV2":"ScanNet","scannetv2":"ScanNet","scannetpp":"ScanNet++","3rscan":"3RScan","multiscan":"MultiScan","roomtour3d":"RoomTour3D","bridgedata_v2":"BridgeData V2"}
LOCATOR_KEYS=("source_qa_id","video_id","scene_id","path_id","clip_id","frame_ids","frame_indices","start_frame","end_frame")
def parse_json(value:Any)->dict[str,Any]:
    if isinstance(value,dict): return value
    if isinstance(value,str):
        try:
            result=json.loads(value); return result if isinstance(result,dict) else {}
        except json.JSONDecodeError: return {}
    return {}
def make_row(row):
    metadata=row.get("metadata") or {}; source=str(metadata.get("source") or metadata.get("dataset") or "")
    if source.startswith("roomtour3d_"): source="RoomTour3D"
    gt=parse_json(metadata.get("gt_json")); locator={key:gt[key] for key in LOCATOR_KEYS if key in gt and gt[key] not in (None,"",[])}
    if not locator and metadata.get("scene"): locator["scene_id"]=metadata["scene"]
    source_name=SOURCE_NAMES.get(source,source)
    hosted_images={"AI2-THOR","ProcTHOR","HSSD","ReplicaCAD","MultiScan","BridgeData V2","SpaceNum","ARKitScenes"}
    hosted_videos={"SIMS-V","ARKitScenes"}
    needs_upstream=(source_name not in (hosted_videos if row["media_type"]=="video" else hosted_images))
    return {"qa_id":row["id"],"source":source_name,"media_type":row["media_type"],"media_count":len(row.get("media_paths") or []),"source_locator":locator,"needs_upstream_media":needs_upstream,"needs_upstream_mapping":not bool(locator)}
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",type=Path,nargs="+",required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); rows=[]
    for path in args.input:
        opener=gzip.open if path.suffix==".gz" else open
        with opener(path,"rt",encoding="utf-8") as handle: rows.extend(make_row(json.loads(line)) for line in handle)
    rows.sort(key=lambda row:row["qa_id"]); args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,"wt",encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps({"rows":len(rows),"output":args.output.name}))
if __name__=="__main__": main()
