if __name__ != "__main__":
    from megatron import print_rank_0
else:
    def print_rank_0(*args, **kwargs): print(*args, **kwargs)

class Symbols:
    MAMBA = 'M'
    ATTENTION = '*'
    MLP = '-'
    VALID = {MAMBA, ATTENTION, MLP}

def allocate_layers(total_layers_count: int, target_attention_ratio: float,
                    target_mlp_ratio: float, override_pattern: str = None):
    assert total_layers_count > 0
    assert target_attention_ratio >= 0.0 and target_attention_ratio <= 1.0
    assert target_mlp_ratio >= 0.0 and target_mlp_ratio <= 1.0
    assert target_attention_ratio + target_mlp_ratio <= 1.0
    # Note: target_mamba_ratio = 1.0 - target_attention_ratio - target_mlp_ratio

    if override_pattern is None:
        # First, allocate attention (evenly spaced, starting and ending with
        # mamba)
        attention_layers_count: int = round(total_layers_count *
                                            target_attention_ratio)
        mamba_layers_count: int = total_layers_count - attention_layers_count
        mamba_sections_count: int = attention_layers_count + 1
        mamba_section_length: float = mamba_layers_count / mamba_sections_count

        layer_type_list = [Symbols.MAMBA] * total_layers_count
        x: float = mamba_section_length
        for l in range(total_layers_count):
            if x < 0.5:
                layer_type_list[l] = Symbols.ATTENTION
                x += mamba_section_length
            else:
                x -= 1

        # Next, allocate mlp
        # (evenly distributed, but right-justified, not replacing attention)
        mlp_layers_count: int = round(total_layers_count * target_mlp_ratio)
        if mlp_layers_count > 0:
            mamba_layers_count -= mlp_layers_count
            mamba_to_mlp_ratio: float = mamba_layers_count / mlp_layers_count

            x: float = mamba_to_mlp_ratio
            for l in range(total_layers_count):
                if layer_type_list[l] == Symbols.MAMBA:
                    if x < 0.5:
                        layer_type_list[l] = Symbols.MLP
                        x += mamba_to_mlp_ratio
                    else:
                        x -= 1
    else:
        print_rank_0("Using hybrid override pattern")
        if target_attention_ratio > 0.0:
            print_rank_0("Warning: overriding target attention ratio of "
                         f"{target_attention_ratio:.2f}")
        if target_mlp_ratio > 0.0:
            print_rank_0("Warning: overriding target mlp ratio of "
                         f"{target_mlp_ratio:.2f}")
        layer_type_list = list(override_pattern)
        override_pattern_length = len(layer_type_list)
        if override_pattern_length != total_layers_count:
            raise ValueError("The hybrid override pattern is the wrong "
                            f"length: got {override_pattern_length}, expected "
                            f"{total_layers_count}")
        for l in layer_type_list:
            if l not in Symbols.VALID:
                raise ValueError(f"In hybrid override pattern, '{l}' is not "
                                 f"one of {Symbols.VALID}")

    if (target_attention_ratio > 0.0 or target_mlp_ratio > 0.0 or
        override_pattern is not None):
        actual_attention_layers_count = layer_type_list.count(Symbols.ATTENTION)
        actual_attention_ratio = (actual_attention_layers_count /
                                  total_layers_count)
        actual_mlp_layers_count = layer_type_list.count(Symbols.MLP)
        actual_mlp_ratio = actual_mlp_layers_count / total_layers_count
        allocation_string = ''.join(layer_type_list)
        print_rank_0(f"Hybrid allocation ({Symbols.MAMBA} is mamba, "
                     f"{Symbols.ATTENTION} is attention, "
                     f"{Symbols.MLP} is mlp):")
        print_rank_0(allocation_string)
        print_rank_0(f"{actual_attention_layers_count} attention layers in "
                     f"{total_layers_count} total layers.")
        print_rank_0(f"Target attention ratio: {target_attention_ratio:.2f}. "
                     f"Actual attention ratio: {actual_attention_ratio:.2f}.")
        print_rank_0(f"{actual_mlp_layers_count} mlp layers in "
                     f"{total_layers_count} total layers.")
        print_rank_0(f"Target mlp ratio: {target_mlp_ratio:.2f}. "
                     f"Actual mlp ratio: {actual_mlp_ratio:.2f}.")

    return layer_type_list

if __name__ == "__main__":
  test_cases = [
      # (10, 0.2, 0.0),
      # (48, 0.0, 0.0), # will not print anything
      # (48, 0.1, 0.0),
      # (48, 0.3, 0.0),
      # (48, 0.5, 0.0),
      # (48, 0.6, 0.0),
      # (48, 0.7, 0.0),
      # (48, 1.0, 0.0),
      # (10, 0.0, 0.1),
      # (10, 0.0, 0.3),
      # (10, 0.0, 0.5),
      # (10, 0.0, 1.0),
      # (10, 0.1, 0.1),
      # (10, 0.2, 0.2),
      # (10, 0.3, 0.3),
      # (10, 0.5, 0.5),
      # (48, 0.2, 0.3),
      (48, 0.5, 0.2),
      # (48, 0.5, 0.2, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
      # (48, 0.0, 0.2, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
      # (48, 0.5, 0.5),
  ]
  for t in test_cases:
    print("")
    allocate_layers(*t)
