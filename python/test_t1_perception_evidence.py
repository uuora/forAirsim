import unittest
from evaluate_t1_ground_perception import identity_score, local_association
from export_t1_balloon_dataset import yolo_label, validate_episode_split


class PerceptionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.truth={"target":{"bbox_xyxy_pixels":[10,10,30,30]},"distractor":{"bbox_xyxy_pixels":[50,10,70,30]}}

    def test_wrong_instance_and_no_output_are_failures(self):
        self.assertEqual(identity_score([50,10,70,30],self.truth)["status"],"WRONG_DISTRACTOR")
        self.assertFalse(identity_score(None,self.truth)["all_attempts_correct"])
        self.assertTrue(identity_score([10,10,30,30],self.truth)["all_attempts_correct"])

    def test_invisible_target_is_not_implicitly_correct(self):
        self.truth["target"]["bbox_xyxy_pixels"]=None
        result=identity_score(None,self.truth)
        self.assertFalse(result["target_visible"])
        self.assertFalse(result["all_attempts_correct"])

    def test_cloud_no_single_box_and_two_local_matches_reject(self):
        self.assertIsNone(local_association([],[[10,10,30,30]])["selected_box"])
        self.assertIsNone(local_association([[10,10,30,30]],[[10,10,30,30],[12,12,30,30]])["selected_box"])
        self.assertEqual(local_association([[10,10,30,30]],[[10,10,30,30]])["status"],"SOURCE_FRAME_LOCAL_ASSOCIATION_ONLY")

    def test_yolo_labels_have_half_open_bbox_dimensions(self):
        self.assertEqual(yolo_label([10,20,30,60],100,100),[0,.2,.4,.2,.4])
        for box in ([30,20,10,60],[0,0,101,10],[0,0,float('nan'),10]):
            with self.assertRaises(ValueError):yolo_label(box,100,100)

    def test_episode_and_duplicate_rgb_cannot_cross_splits(self):
        rows=[{"episode_id":"a","rgb_sha256":"1","split":"train"},
              {"episode_id":"b","rgb_sha256":"2","split":"validation"},
              {"episode_id":"c","rgb_sha256":"3","split":"test"}]
        self.assertEqual(validate_episode_split(rows),{"train":1,"validation":1,"test":1})
        for key in ("episode_id","rgb_sha256"):
            changed=[dict(row) for row in rows];changed[2][key]=changed[0][key]
            with self.assertRaises(ValueError):validate_episode_split(changed)

    def test_development_only_cannot_claim_train_test_readiness(self):
        with self.assertRaises(ValueError):validate_episode_split([{"episode_id":"a","rgb_sha256":"1","split":"DEVELOPMENT_UNPARTITIONED"}])


if __name__=="__main__":unittest.main()
