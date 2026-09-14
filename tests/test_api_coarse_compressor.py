import ast
import unittest

from api_coarse_compressor import (
    APIRecommendationCoarseCompressor,
    build_compressed_api_prompt,
    extract_code_from_api_prompt,
)


class FakeScorer:
    """Deterministic scorer: preserves the compressor interface without model downloads."""
    def get_token_length(self, text):
        return len(text.split())

    def get_condition_ppl(self, text, question, condition_in_question="prefix"):
        # Give a predictable but distinct AMI value to every unit.
        return float(len(set(text.split()) & set(question.split())))


SOURCE = '''import os


def create_model():
    model = Model()
    return model


class Runner:
    @staticmethod
    def irrelevant():
        return os.getcwd()

    def train(self):
        model = create_model()
        model.[MASK]()


def unused():
    return 42
'''

PROMPT = "你是助手。\n请仅输出 5 行 API 签名，每行一个，不要输出编号或其他解释性文字。\n" + SOURCE + "推荐的 5 个 API 签名："


class APIRecommendationCoarseCompressorTest(unittest.TestCase):
    def setUp(self):
        self.compressor = APIRecommendationCoarseCompressor(FakeScorer())

    def test_keeps_mask_and_direct_data_flow_dependency(self):
        result = self.compressor.compress(SOURCE, token_budget=1000)
        self.assertTrue(result["syntax_check"]["valid"])
        self.assertIn("Runner.train", result["mask_unit"])
        self.assertTrue(any("create_model" in item for item in result["required_dependency_units"]))
        self.assertIn("def create_model", result["compressed_code"])
        self.assertIn("def train", result["compressed_code"])
        self.assertIn("model.[MASK]()", result["compressed_code"])
        detail = next(x for x in result["candidate_units"] if x["qualified_name"] == "create_model")
        self.assertGreater(detail["adf_if"], 0)
        self.assertIn("ami_rank", detail)
        self.assertIn("fused_score", detail)
        ast.parse(result["compressed_code"].replace("[MASK]", "__MASK__"))

    def test_reuses_analysis_when_only_budget_changes(self):
        first = self.compressor.compress(SOURCE, token_budget=1000)
        second = self.compressor.compress(SOURCE, token_budget=1)
        self.assertEqual(first["mask_unit"], second["mask_unit"])
        self.assertGreater(second["budget_exceeded_by_mandatory_units"], 0)
        self.assertTrue(second["syntax_check"]["valid"])

    def test_requires_exactly_one_mask_inside_a_unit(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.compressor.analyze("def f():\n    return 1\n")
        with self.assertRaisesRegex(ValueError, "inside a function"):
            self.compressor.analyze("x = [MASK]\ndef f():\n    return x\n")

    def test_dataset_prompt_is_extracted_and_rebuilt_without_wrapper_in_code(self):
        prefix, code, suffix = extract_code_from_api_prompt(PROMPT)
        self.assertTrue(prefix.endswith("文字。\n"))
        self.assertEqual(code, SOURCE.rstrip())
        self.assertEqual(suffix, "推荐的 5 个 API 签名：")
        result = self.compressor.compress_dataset(
            [{"id": 9, "gt": "run", "prompt": PROMPT}, {"id": 10, "gt": "x", "prompt": "bad"}], 1000)
        self.assertEqual(result["compressed_records"], 1)
        self.assertEqual(result["skipped_records"], 1)
        first = result["records"][0]
        self.assertEqual(first["status"], "ok")
        self.assertEqual(build_compressed_api_prompt(prefix, first["coarse_result"]["compressed_code"], suffix), first["compressed_prompt"])
        self.assertEqual(result["records"][1]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
