"""Hash-bound supplemental clock evidence must not upgrade a failed session."""
import json
from pathlib import Path
from test_t1_qt_workbench import WorkbenchTests
from t1_qt_dashboard import ExperimentDashboard
from t1_qt_evidence import load_campaign,sha256


class ProcessQtTests(WorkbenchTests):
    def supplement(self):
        data=json.loads(self.analysis.read_text());source=data['runs'][0]
        source['base_evidence']['issues']=[dict(issue='TARGET_EXPLICIT_CLEAR_AFTER_OBSERVATION_DEADLINE')]
        source['independent_scale_qualification']='FAIL_SCALED_MOTION_EVIDENCE'
        source['architecture']='SPAWN_PROCESS_OWN_STATE'
        original=self.analysis.parent/'supplement_original.json';self.write(original,data)
        original_report=Path(source['run'])/'flight/report.json'
        supplement=self.analysis.parent/'clear_supplement.json'
        self.write(supplement,dict(kind='T1_INTERRUPTED_TARGET_CLEAR_SUPPLEMENT',session_id=source['session_id'],
            supplemental_only=True,overall_experiment_qualification_changed=False,
            status='PASS_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION',
            input_receipts=[dict(path=str(original_report),sha256=sha256(original_report))],
            rows=[dict(actual_first_clear_within_original_deadline=True,actual_first_clear_margin_s=.184)]))
        source['interrupted_target_clear_supplement']=dict(path=str(supplement),sha256=sha256(supplement))
        self.write(self.analysis,data)
        return supplement,original_report

    def test_supplement_displays_both_scopes_and_preserves_failure(self):
        self.supplement()
        dashboard=ExperimentDashboard();self.widgets.append(dashboard);dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        self.assertEqual(dashboard.run.evidence_status,'FAIL_SCALED_MOTION_EVIDENCE')
        self.assertIn('V11控制进程',dashboard.run.label)
        self.assertIn('原清零晚到提示保留',dashboard.status_label.text())
        self.assertIn('提前184毫秒',dashboard.status_label.text())
        self.assertIn('收尾窗口另计',dashboard.status_label.text())

    def test_modified_supplement_refused(self):
        supplement,_=self.supplement();supplement.write_text('{}')
        with self.assertRaisesRegex(ValueError,'校验值'):load_campaign(self.analysis)

    def test_modified_original_input_refused(self):
        _,original=self.supplement();original.write_text('{}')
        with self.assertRaisesRegex(ValueError,'原始输入'):load_campaign(self.analysis)

for _name in dir(WorkbenchTests):
    if _name.startswith('test_') and _name not in ProcessQtTests.__dict__:
        setattr(ProcessQtTests,_name,None)
