import csv
import tempfile
import unittest
from pathlib import Path

from common import digest
from reviews import import_reviews


class ReviewTests(unittest.TestCase):
    def package(self,path, verdict="可接受", response="answer"):
        with path.open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["blind_id","instruction","context","response","reference","verdict","reason"])
            writer.writerow(["R001","question","context",response,"reference",verdict,"reviewed"])
        return {"identity":"fixed","items":[{"blind_id":"R001","source_index":17,
                  "content_digest":digest(["R001","question","context","answer","reference"])}]}

    def test_verdict_is_explicit_and_bound_to_response_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"review.csv"
            key=self.package(path)
            result=import_reviews(path,key,"reviewer","human")
            self.assertEqual(result["labels"],{17:"good"})
            self.assertEqual(result["reviewer_type"],"human")
            self.package(path,verdict="")
            with self.assertRaises(ValueError):
                import_reviews(path,key,"reviewer","human")
            self.package(path,response="different model response")
            with self.assertRaises(ValueError):
                import_reviews(path,key,"reviewer","human")


if __name__=="__main__":
    unittest.main()
