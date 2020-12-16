# Experimental support for tensor core

This mainly contains 2 commits:
1. Add a TensorizeStep for AutoScheduler
2. Add experimental support for tensor core sketch rules

See the git diff for more information.

And these UTs may help you better understand how to use.
```
incubator-tvm/tests/python/unittest/test_auto_scheduler_sketch_generation.py
incubator-tvm/tests/python/unittest/test_auto_scheduler_search_policy.py
incubator-tvm/tests/python/unittest/test_auto_scheduler_loop_state.py
```

# Known issues

1. All steps in Auto Schedule needs to be serialized to log, so I put a registered function name string in TensorizeStep, which will be called when applying the step. We may consider to have a better implementation for this.
2. Currently there is a bug with tvm::support::parallel_for calling a python function in its sub threads, so I make a hack on `incubator-tvm/src/support/parallel_for.cc`. To fix this, we need to move all the functions defined in `python/tvm/auto_scheduler/test_sketch/test_tensor_core_sketch.py` to C++.
3. Currently Auto Scheudle's fill tile size part does not support to set some of the split factors to a fixed number( and evolutionary search also does not ...), so I used a different compute expression with input to be (N // 16, N // 16, 16, 16) in `tests/python/unittest/test_auto_scheduler_common.py`. To fix this, we may need to more modifications.
4. Currently the tensor core sketch rule is really simple, just skip or apply with skipping the rest rules, this can be better designed to cooperate with other rules.
