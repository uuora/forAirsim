"""Derived reports preserve raw evidence and accept relative campaign paths."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .campaign import PLANNER_CONFIG
from .comparison import render


class ComparisonTests(unittest.TestCase):
    def test_overlapping_records_have_distinct_styles_and_independent_rows(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            folder=Path(tmp);results=[];hashes={}
            (folder/'config_frozen.json').write_bytes(PLANNER_CONFIG.read_bytes())
            rows=[dict(phase='TRACKING',t=t,position=[t,0,-5],velocity=[.8,0,0],
                       reference=dict(p=[t,0,-5]),attitude_reference_error_rad=0.) for t in (.1,.2)]
            for method in ('bezier','hermite','bspline'):
                flight=folder/'flights'/method;flight.mkdir(parents=True)
                samples=flight/'samples.jsonl'
                samples.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                hashes[samples]=hashlib.sha256(samples.read_bytes()).hexdigest()
                results.append(dict(method=method,run_dir=str(flight)))
            (folder/'summary.json').write_text(json.dumps(dict(results=results,comparable_conditions=True)),encoding='utf-8')
            (folder/'comparison.csv').write_text('method\nbezier\nhermite\nbspline\n',encoding='utf-8')
            with patch('matplotlib.pyplot.close',wraps=plt.close) as closed:
                page=render(folder)
            figures=[call.args[0] for call in closed.call_args_list if call.args and hasattr(call.args[0],'axes')]
            self.assertEqual(len(figures),3)
            for ax in figures[0].axes:
                lines=[line for line in ax.lines if line.get_label() in ('bezier','hermite','bspline')]
                self.assertEqual({line.get_label() for line in lines},{'bezier','hermite','bspline'})
                self.assertEqual(len({line.get_linestyle() for line in lines}),3)
                self.assertEqual(len({line.get_marker() for line in lines}),3)
                for line in lines:
                    x=line.get_data_3d()[0] if hasattr(line,'get_data_3d') else line.get_xdata()
                    self.assertEqual(list(x),[.1,.2])
            self.assertEqual(len(figures[1].axes),12)
            for index,method in enumerate(('bezier','hermite','bspline')):
                self.assertEqual(figures[1].axes[index*4].get_title(),method)
                self.assertEqual(list(figures[1].axes[index*4+1].lines[0].get_ydata()),[0.,0.])
            self.assertIn('by_method.png',page.read_text(encoding='utf-8'))
            self.assertIn('trajectory_3d.png',page.read_text(encoding='utf-8'))
            framing=json.loads((folder/'trajectory_3d_view.json').read_text(encoding='utf-8'))
            self.assertEqual(framing['default_mode'],'display')
            self.assertEqual(framing['display_box_ratio'],[1.55,1.1,.8])
            self.assertEqual(framing['samples_by_method'],dict(bezier=2,hermite=2,bspline=2))
            self.assertTrue((folder/'trajectory_3d_metric.png').is_file())
            spatial_lines=[line for line in figures[2].axes[0].lines
                           if line.get_label() in ('bezier','hermite','bspline')]
            self.assertEqual(len(spatial_lines),3)
            for line in spatial_lines:self.assertEqual(list(line.get_data_3d()[0]),[.1,.2])
            provenance=json.loads((folder/'plot_provenance.json').read_text(encoding='utf-8'))
            self.assertIn('by_method.svg',provenance['outputs'])
            for samples,before in hashes.items():self.assertEqual(hashlib.sha256(samples.read_bytes()).hexdigest(),before)

    def test_relative_path_report_has_bound_plot_source_and_preserves_samples(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            folder=Path(tmp);flight=folder/'flights'/'synthetic';flight.mkdir(parents=True)
            (folder/'config_frozen.json').write_bytes(PLANNER_CONFIG.read_bytes())
            samples=flight/'samples.jsonl'
            rows=[dict(phase='TRACKING',t=t,position=[t,0,-5],velocity=[.8,0,0],
                       reference=dict(p=[t,0,-5]),attitude_reference_error_rad=0.) for t in (.1,.2)]
            samples.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            before=hashlib.sha256(samples.read_bytes()).hexdigest()
            (folder/'summary.json').write_text(json.dumps(dict(results=[dict(method='bezier',run_dir=str(flight),
                normal_completion=False,reference_time_s=1.)],comparable_conditions=False)),encoding='utf-8')
            (folder/'comparison.csv').write_text('method\nbezier\n',encoding='utf-8')
            page=render(os.path.relpath(folder))
            self.assertIn('flights/synthetic/report.json',page.read_text(encoding='utf-8'))
            provenance=json.loads((folder/'plot_provenance.json').read_text(encoding='utf-8'))
            self.assertEqual(provenance['inputs'][str(samples)],before)
            self.assertEqual(hashlib.sha256(samples.read_bytes()).hexdigest(),before)
            self.assertTrue((folder/'comparison.png').is_file())
            self.assertTrue((folder/'plot_source'/'comparison.py').is_file())
