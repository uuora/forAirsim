"""Small functional and corruption checks; no 24-rollout campaign is run."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_control_simulation_mvp import expected_coefficients, polynomial, verify_basis, verify_config, verify_record, verify_samples
from run_control_simulation_mvp import basis_check, evaluate_reference, make_config, run, SAMPLE_FIELDS, sample_rows, write_csv


class ControlSimulationMVPTests(unittest.TestCase):
    def test_frozen_matrix_is_six_references_times_four_single_factor_conditions(self):
        config=make_config()
        verify_config(config)
        self.assertEqual(len(set(config["frozen_execution_order"])),24)
        config["conditions"][0]["velocity_lag_s"] = .45
        with self.assertRaises(AssertionError):
            verify_config(config)

    def test_basis_checks_and_independent_endpoints(self):
        config=make_config()
        checks=[basis_check(ref,config) for ref in config["references"]]
        verify_basis(checks,config)
        for ref in config["references"]:
            c=expected_coefficients(ref["parameters"],config["boundary"])
            duration=ref["parameters"][-1]
            for derivative,start,end in ((0,[0.,0.,5.],[8.,0.,5.]),(1,[1.,0.,0.],[1.,0.,0.]),(2,[0.,0.,0.],[0.,0.,0.])):
                np.testing.assert_allclose(polynomial(c,[0.,duration],duration,derivative),[start,end],atol=1e-12,rtol=0)

    def test_nominal_record_replays_and_terminal_csv_has_no_invented_command(self):
        config=make_config()
        record=evaluate_reference(config["references"][0],config["conditions"][0],config)
        verify_record(record,config["references"][0],config["conditions"][0],config)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"samples.csv"
            write_csv(path,SAMPLE_FIELDS,sample_rows(record["trace"]))
            verify_samples(path,record)
            with path.open(encoding="utf-8",newline="") as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(rows[-1]["cmd_mps_x"],"")
            self.assertEqual(rows[-1]["command_interval_s"],"")

    def test_perturbed_record_replays_and_state_corruption_is_detected(self):
        config=make_config()
        reference,spec=config["references"][3],config["conditions"][1]
        record=evaluate_reference(reference,spec,config)
        verify_record(record,reference,spec,config)
        tampered=copy.deepcopy(record)
        tampered["trace"]["actual_position_m"][10][1]+=.01
        with self.assertRaises(AssertionError):
            verify_record(tampered,reference,spec,config)

    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory),make_plots=False)


if __name__=="__main__":
    unittest.main()
