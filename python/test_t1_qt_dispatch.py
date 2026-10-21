"""New saved dispatch evidence stays explicit in Qt; no simulator/API use."""
import json
from test_t1_qt_workbench import WorkbenchTests
from t1_qt_dashboard import ExperimentDashboard


class DispatchQtTests(WorkbenchTests):
    # Reuse only setup/helpers, without collecting the parent's test methods.
    def test_new_revision_label_and_recovery_counts_are_visible(self):
        data=json.loads(self.analysis.read_text()); source=data['runs'][0]
        source['architecture']='INDEPENDENT_CONTROL_OWN_STATE_BOUNDED_PERSISTENCE_V10P2'
        source['feedback_dispatch_audit']=dict(status='PASS_STRICT_FEEDBACK_DISPATCH_AUDIT',refused_attempts=2,recovered_ticks=2)
        self.write(self.analysis,data)
        dashboard=ExperimentDashboard();self.widgets.append(dashboard);dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        self.assertIn('V10P2',dashboard.run.label)
        self.assertIn('过期拒绝2次',dashboard.status_label.text())
        self.assertIn('新状态恢复2次',dashboard.status_label.text())
        self.assertIn('120毫秒',dashboard.status_label.text())

    def test_failed_dispatch_audit_stays_failed_when_runner_metrics_look_good(self):
        data=json.loads(self.analysis.read_text());data['runs'][0]['feedback_dispatch_audit']=dict(status='FAIL_FEEDBACK_DISPATCH_EVIDENCE')
        self.write(self.analysis,data)
        dashboard=ExperimentDashboard();self.widgets.append(dashboard);dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        self.assertIn('反馈时效核查未通过',dashboard.status_label.text())

    def test_late_target_clear_is_visible_even_with_all_correct_images(self):
        data=json.loads(self.analysis.read_text())
        data['runs'][0]['base_evidence']['issues']=[dict(issue='TARGET_EXPLICIT_CLEAR_AFTER_OBSERVATION_DEADLINE')]
        self.write(self.analysis,data)
        dashboard=ExperimentDashboard();self.widgets.append(dashboard);dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        self.assertIn('实际清零超过源观测截止时间',dashboard.status_label.text())


# unittest would otherwise rerun all inherited frozen workbench tests here.
for _name in dir(WorkbenchTests):
    if _name.startswith('test_') and _name not in DispatchQtTests.__dict__:
        setattr(DispatchQtTests,_name,None)
