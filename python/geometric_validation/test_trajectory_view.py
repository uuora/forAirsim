"""Interactive artifacts retain exact recorded samples and work without remote assets."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from .campaign import PLANNER_CONFIG
from .trajectory_view import campaign_view, write_view


class ViewTests(unittest.TestCase):
    def test_offline_export_preserves_samples_and_escapes_labels(self):
        class Page(HTMLParser):
            def __init__(self):super().__init__();self.payload='';self.reading=False;self.remote=[];self.images=[]
            def handle_starttag(self,tag,attrs):
                values=dict(attrs)
                if tag=='script':
                    self.reading=values.get('id')=='trajectory-data'
                    if 'src' in values:self.remote.append(values['src'])
                if tag=='img':self.images.append(values)
            def handle_endtag(self,tag):
                if tag=='script':self.reading=False
            def handle_data(self,data):
                if self.reading:self.payload+=data
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);flight=folder/'flight';flight.mkdir()
            (folder/'config_frozen.json').write_bytes(PLANNER_CONFIG.read_bytes())
            samples=flight/'samples.jsonl'
            rows=[dict(phase='TRACKING',t=t,position=[t,2,-5],reference=dict(p=[t,1,-4])) for t in (.1,.217,.31)]
            rows.insert(0,dict(phase='PREPARING',position=[0,0,0]))
            samples.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            before=hashlib.sha256(samples.read_bytes()).hexdigest()
            (folder/'summary.json').write_text(json.dumps(dict(results=[dict(method='bezier',run_dir=str(flight))])),encoding='utf-8')
            malicious='</script><img src="remote">'
            view=campaign_view(folder,'fixture',malicious)
            page=write_view(folder,[view],malicious);parser=Page();parser.feed(page.read_text(encoding='utf-8'))
            data=json.loads(parser.payload)['views'][0];series=data['series'][0]
            self.assertEqual(data['label'],malicious)
            self.assertEqual(series['t'],[.1,.217,.31])
            self.assertEqual(series['measured'],[[.1,2,5],[.217,2,5],[.31,2,5]])
            self.assertEqual(series['reference'],[[.1,1,4],[.217,1,4],[.31,1,4]])
            self.assertEqual(parser.remote,[]);self.assertEqual(parser.images,[])
            self.assertEqual(hashlib.sha256(samples.read_bytes()).hexdigest(),before)
            provenance=json.loads((folder/'trajectory_view_provenance.json').read_text(encoding='utf-8'))
            self.assertEqual(provenance['inputs'][str(samples)],before)
            self.assertEqual(provenance['samples'],dict(fixture=dict(bezier=3)))
            self.assertEqual(provenance['output_sha256'],hashlib.sha256(page.read_bytes()).hexdigest())

    def test_missing_record_has_no_interactive_flight_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(campaign_view(tmp))
            self.assertIsNone(write_view(tmp,[None]))
            self.assertFalse((Path(tmp)/'trajectory_view.html').exists())
