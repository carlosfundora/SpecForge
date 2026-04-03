import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaForCausalLMEagle3,
    LlamaMLP,
    LlamaRMSNorm,
)

# from model_module import LlamaForCausalLMEagle3


class TestLlamaForCausalLMEagle3Loading(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()

        config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 128256,
            "draft_vocab_size": 32000,
            "parallel_drafting": False,
            "k_train": 8,
            "cod_retention": 0.8,
            "mask_token_id": 0,
        }

        self.config = LlamaConfig(**config_dict)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        model = LlamaForCausalLMEagle3(self.config)

        self.assertIsInstance(model.midlayer.self_attn, LlamaAttention)
        self.assertIsInstance(model.midlayer.mlp, LlamaMLP)
        self.assertIsInstance(model.midlayer.hidden_norm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.input_layernorm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.post_attention_layernorm, LlamaRMSNorm)
        self.assertEqual(model.midlayer.hidden_size, self.config.hidden_size)

    def test_save_pretrained(self):
        """Test the model's save_pretrained functionality."""
        model = LlamaForCausalLMEagle3(self.config)

        self.config.save_pretrained(self.temp_dir)

        model_path = os.path.join(self.temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(model_path))

    @patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    def test_from_pretrained_mock(self, mock_from_pretrained):
        """mock"""
        mock_model = LlamaForCausalLMEagle3(self.config)
        mock_from_pretrained.return_value = mock_model

        loaded_model = LlamaForCausalLMEagle3.from_pretrained(self.temp_dir)
        mock_from_pretrained.assert_called_once_with(self.temp_dir)
        self.assertIsInstance(loaded_model, LlamaForCausalLMEagle3)

    def test_model_forward_pass(self):
        """forward"""
        model = LlamaForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 2
        seq_len = 10

        input_emb = torch.randn(batch_size, seq_len, self.config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=input_emb,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        self.assertEqual(outputs.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_state_dict_compatibility(self):
        model1 = LlamaForCausalLMEagle3(self.config)
        model2 = LlamaForCausalLMEagle3(self.config)

        state_dict = model1.state_dict()

        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    def test_config_validation(self):
        invalid_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=127,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        with self.assertRaises(AttributeError):
            LlamaForCausalLMEagle3(invalid_config)

    def test_prepare_p_eagle_inputs(self):
        parallel_config = LlamaConfig(
            **{
                **self.config.to_dict(),
                "parallel_drafting": True,
                "mask_token_id": 17,
            }
        )
        model = LlamaForCausalLMEagle3(parallel_config)
        model.mask_hidden.data.fill_(0.25)

        last_token_ids = torch.tensor([[5], [9]], dtype=torch.long)
        fused_hidden_states = torch.randn(2, 1, model.fc.in_features)
        hidden_states, input_embeds, attention_mask, position_ids = (
            model.prepare_p_eagle_inputs(
                last_token_ids=last_token_ids,
                fused_hidden_states=fused_hidden_states,
                k=4,
            )
        )

        expected_hidden_inputs = torch.cat(
            [
                fused_hidden_states,
                model.mask_hidden.expand(2, 3, -1).to(fused_hidden_states.dtype),
            ],
            dim=1,
        )
        expected_input_ids = torch.tensor(
            [[5, 17, 17, 17], [9, 17, 17, 17]], dtype=torch.long
        )

        self.assertEqual(hidden_states.shape, (2, 4, parallel_config.hidden_size))
        self.assertEqual(input_embeds.shape, (2, 4, parallel_config.hidden_size))
        self.assertTrue(torch.equal(attention_mask, torch.ones(2, 4, dtype=torch.bool)))
        self.assertTrue(
            torch.equal(position_ids, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
        )
        self.assertTrue(
            torch.allclose(
                hidden_states,
                model.project_hidden_states(expected_hidden_inputs),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                input_embeds,
                model.embed_input_ids(expected_input_ids),
                atol=1e-5,
            )
        )

    def test_parallel_drafting_round_trip(self):
        config_dict = {
            **self.config.to_dict(),
            "architectures": ["LlamaForCausalLMEagle3"],
            "parallel_drafting": True,
            "mask_token_id": 23,
            "k_train": 6,
            "cod_retention": 0.7,
        }
        config_path = os.path.join(self.temp_dir, "config.json")
        with open(config_path, "w") as f:
            import json

            json.dump(config_dict, f)

        loaded_config = LlamaConfig.from_pretrained(self.temp_dir)
        self.assertTrue(loaded_config.parallel_drafting)
        self.assertEqual(loaded_config.mask_token_id, 23)
        self.assertEqual(loaded_config.k_train, 6)
        self.assertAlmostEqual(loaded_config.cod_retention, 0.7)

        model = LlamaForCausalLMEagle3(loaded_config)
        model.mask_hidden.data.fill_(1.5)
        model.save_pretrained(self.temp_dir)
        reloaded = LlamaForCausalLMEagle3.from_pretrained(self.temp_dir)

        self.assertTrue(reloaded.parallel_drafting)
        self.assertEqual(reloaded.mask_token_id, 23)
        self.assertTrue(torch.allclose(reloaded.mask_hidden, model.mask_hidden))


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestLlamaForCausalLMEagle3Loading))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
