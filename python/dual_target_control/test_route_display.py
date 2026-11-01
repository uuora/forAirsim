"""Landmark evidence, stable metre projection, and idempotent UE viewing."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock,patch
import numpy as np
from .route_milestones import RouteMilestones
from .route_curve import route_spec


class LandmarkTests(TestCase):
    def test_only_contacted_and_departed_midpoint_is_visited(self):
        spec=route_spec([[8,0,5],[12,6,7],[6,10,6]])
        tracker=RouteMilestones(spec);times=[dict(arrival_time_s=5),dict(arrival_time_s=10),dict(arrival_time_s=15)]
        self.assertEqual([m['kind'] for m in tracker.markers],['start','via','via','goal'])
        tracker.update(5,[8.1,0,5],0,times)
        self.assertEqual(tracker.markers[1]['state'],'pending')
        tracker.update(6,[9,0,5],1,times)
        self.assertEqual(tracker.markers[1]['state'],'visited')
        tracker.update(11,[20,20,5],2,times)
        self.assertEqual(tracker.markers[2]['state'],'unconfirmed')
        self.assertEqual(RouteMilestones(spec).events,[])
        self.assertEqual(RouteMilestones(spec).markers[1]['state'],'pending')

    def test_pending_via_has_no_ue_pass_marker(self):
        from .overview import OverviewStream
        from .core import ControlConfig
        from .test_overview import FakeClient
        import airsim
        class PlotClient(FakeClient):
            def simPlotStrings(self,*args,**kwargs):self.calls.append(('simPlotStrings',args,kwargs))
        cfg=dict(control={},overview=dict(ue_debug_markers=True))
        runtime=Mock();stream=OverviewStream(cfg,runtime);stream._debug_markers=True
        marks=RouteMilestones(route_spec([[8,0,5],[12,6,7]])).markers
        state=dict(config=cfg,targets=[],planned_path=[],trail=[],route_markers=marks,marker_world_origin_ned=[10,20,30])
        c=PlotClient();stream._draw(c,airsim,state)
        texts=[call for call in c.calls if call[0]=='simPlotStrings']
        self.assertEqual(texts[0][1][0],['S','E'])
        marks[1]['state']='visited';c=PlotClient();stream._draw(c,airsim,state)
        points=[call for call in c.calls if call[0]=='simPlotPoints']
        self.assertEqual(len(points),1)
        self.assertEqual(points[0][1][0][0].to_numpy_array().tolist(),[18.,20.,25.])


class StableSceneTests(TestCase):
    def scene(self):
        from .native_scene import NativeScene
        from prepared_route_panel import prepared_config
        runtime=Mock();runtime.snapshot.return_value=dict(mode='airsim',connected=True)
        scene=NativeScene(prepared_config(),runtime);scene._client=Mock();scene._window=Mock()
        scene._window.show.return_value=dict(pid=7);scene._window.hide.return_value=dict(pid=7)
        return scene

    def test_repeated_show_preserves_camera_and_foreground(self):
        scene=self.scene()
        with patch.object(scene,'_initialize'),patch('dual_target_control.native_scene.reflected_command',side_effect=lambda _,command,*a:command) as rpc:
            scene._switch(True);initial=rpc.call_count
            commands=[c.args[1] for c in rpc.call_args_list]
            self.assertTrue(any(c.endswith('ResetOverview') for c in commands))
            self.assertFalse(any('SetFollow true' in c for c in commands))
            self.assertLess(next(i for i,c in enumerate(commands) if c.endswith('SetEnabled true')),next(i for i,c in enumerate(commands) if 'SetEnableWorldRendering' in c))
            scene._switch(True);self.assertEqual(rpc.call_count,initial);scene._window.show.assert_called_once()
            scene._switch(False);scene._switch(True)
            commands=[c.args[1] for c in rpc.call_args_list]
            self.assertEqual(sum(c.endswith('ResetOverview') for c in commands),1)

    def test_task_configure_reuses_scene_without_closing_it(self):
        from .prepared_route_live import PreparedRouteRuntime
        from prepared_route_panel import prepared_config
        import time
        cfg=prepared_config();runtime=PreparedRouteRuntime(cfg,mode='demo');scene=Mock();runtime._scene=scene
        try:
            new=deepcopy(cfg);new['preparation']['task']['start']=[-3,1,5]
            runtime.configure(new);deadline=time.monotonic()+3
            while not scene.update_config.called and time.monotonic()<deadline:time.sleep(.01)
            self.assertIs(runtime._scene,scene);scene.close.assert_not_called();scene.update_config.assert_called_once()
        finally:runtime.shutdown();runtime.wait_closed(5)

    def test_inflight_target_updates_enqueue_without_control_rpc(self):
        from .prepared_route_live import PreparedRouteRuntime
        from prepared_route_panel import prepared_config
        from types import SimpleNamespace
        from .route_curve import route_spec
        runtime=PreparedRouteRuntime.__new__(PreparedRouteRuntime)
        runtime._active=True;runtime.mode='airsim';runtime.route_index=1;runtime.marker_origin=np.zeros(3)
        runtime.marker_cache=None;runtime._run_dir='test_run';runtime.live_elapsed=6.;runtime._scene=Mock()
        runtime._client=Mock();runtime._publish=Mock()
        runtime.live_curve=SimpleNamespace(spec=route_spec([[8,0,5],[12,6,7],[6,10,6]]))
        runtime._markers(True)
        runtime._client.assert_not_called();self.assertEqual(runtime._client.method_calls,[])
        payload=runtime._scene.sync_targets.call_args.args[0]
        self.assertEqual(payload['points'],[[12,6,7],[6,10,6]])
        self.assertEqual(payload['reference_time_s'],6.)
        runtime._markers(True);runtime._scene.sync_targets.assert_called_once()


class ProjectionTests(TestCase):
    def test_route_view_scale_and_camera_stay_fixed_as_drone_moves(self):
        from PyQt5.QtWidgets import QApplication
        from .route_view import RouteFlightView
        from prepared_route_panel import prepared_config
        app=QApplication.instance() or QApplication([]);cfg=prepared_config();view=RouteFlightView(cfg)
        for width,height in ((615,274),(1200,650)):
            view.resize(width,height);view.reset_view();view._camera();matrix=view.projection_matrix.copy()
            projection=view.project_points(view.corners)
            self.assertGreater(projection[:,0].min(),20);self.assertLess(projection[:,0].max(),width-10)
            self.assertGreater(projection[:,1].min(),25);self.assertLess(projection[:,1].max(),height-25)
            view.set_snapshot(dict(config=cfg,position=[12,6,7],trail=[[0,0,5],[12,6,7]]));view._camera()
            np.testing.assert_allclose(view.projection_matrix,matrix)
            original_snapshot=deepcopy(view.snapshot)
            for theme in ('dark','light'):
                view.scene.set_theme(theme);view._camera()
                np.testing.assert_allclose(view.projection_matrix,matrix)
                self.assertEqual(view.snapshot,original_snapshot)
            for plane,axes in (('xy',[[1,0,0],[0,1,0]]),('xz',[[1,0,0],[0,0,1]])):
                view.set_plane(plane);view._camera();lengths=np.linalg.norm(np.asarray(axes)@view.projection_matrix,axis=1)
                self.assertAlmostEqual(*lengths,places=10)
        view.close()
