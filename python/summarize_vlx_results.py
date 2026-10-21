"""Generate summary report from VLX API test results."""
import json
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

result_path = Path("logs/vlx_api_full_test.json")
data = json.loads(result_path.read_text())

print("=" * 70)
print("VLX-Seek API 海事图像回放测试报告")
print("=" * 70)
print()

# Overall stats
print(f"📊 总体统计")
print(f"  状态: {data['status']}")
print(f"  总帧数: {len(data['frames'])}")
print(f"  模型: vlx-seek-1.5-3b (云端 API)")
print(f"  提案模式: {data['proposal_mode']}")
print()

# Summary by profile
summary = data.get("summary_by_profile", {})
if summary:
    print("📈 按条件统计")
    print()
    print("条件             | 帧数 | IoU≥0.1 | IoU≥0.5 | YOLO最高 | 最右规则 | VLX-API")
    print("-" * 70)
    for profile, stats in sorted(summary.items()):
        print(f"{profile:15s} | {stats['frames']:4d} | "
              f"{stats['target_proposal_iou_ge_0_1']:7d} | "
              f"{stats['target_proposal_iou_ge_0_5']:7d} | "
              f"{stats['yolo_top_balloon_target']:8d} | "
              f"{stats['rightmost_balloon_target']:8d} | "
              f"{stats.get('vlx_target', 0):7d}")
    print()

# Detailed analysis
total_frames = len(data['frames'])
vlx_correct = 0
vlx_other = 0
vlx_no_selection = 0
large_deviations = []

for frame in data['frames']:
    vlx_eval = frame.get('vlx_selection_evaluation', {})
    association = vlx_eval.get('association')

    if association == 'TARGET':
        vlx_correct += 1
    elif association == 'OTHER_INSTANCE':
        vlx_other += 1
    elif association == 'NO_SELECTION':
        vlx_no_selection += 1

    # Check for large deviations (in target_occluded frames)
    if frame['profile'] == 'target_occluded':
        vlx_result = frame.get('vlx_result', {})
        if vlx_result and vlx_result.get('result_bbox_list'):
            vlx_box = vlx_result['result_bbox_list'][0]
            proposals = frame.get('proposals', [])

            # Calculate deviation from closest proposal
            vlx_xyxy = [vlx_box['xmin'], vlx_box['ymin'], vlx_box['xmax'], vlx_box['ymax']]
            min_diff = float('inf')
            for p in proposals:
                diff = sum(abs(a - b) for a, b in zip(vlx_xyxy, p['bbox_xyxy']))
                min_diff = min(min_diff, diff)

            if min_diff > 10:
                large_deviations.append({
                    'frame': frame['frame'],
                    'deviation': min_diff,
                    'association': association
                })

print(f"🎯 VLX-Seek API 性能")
print(f"  正确选中目标: {vlx_correct}/{total_frames} ({vlx_correct/total_frames*100:.1f}%)")
print(f"  选中其他实例: {vlx_other}/{total_frames}")
print(f"  未选择: {vlx_no_selection}/{total_frames}")
print()

if large_deviations:
    print(f"⚠️  遮挡条件下的大偏差 (>10 像素)")
    for item in large_deviations:
        print(f"  {item['frame']}: 偏差 {item['deviation']:.1f} 像素 -> {item['association']}")
    print()

# Token usage (if available)
total_tokens = 0
for frame in data['frames']:
    vlx_result = frame.get('vlx_result', {})
    # Note: API response structure doesn't include usage in frame-level results
    # This would need to be tracked during execution

print("💰 成本估算")
print(f"  预计总 tokens: ~{total_frames * 460} (基于测试平均)")
print(f"  预计成本: 取决于平台定价")
print()

print("=" * 70)
print("✅ 测试完成！结果已保存至: logs/vlx_api_full_test.json")
print("=" * 70)
