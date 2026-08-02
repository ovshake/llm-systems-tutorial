(llm-systems-tutorial) abhishek.maiti@dgx-97:~/projects/llm-systems-tutorial/chapters/02-memory-compute-budget$ python measure_memory.py
/home/abhishek.maiti/projects/llm-systems-tutorial/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
preset      predicted     measured    delta
--------------------------------------------
tiny       636.47 MiB     6.03 GiB  +869.8%
small        2.54 GiB    10.23 GiB  +302.3%
medium       7.14 GiB    19.67 GiB  +175.4%
1B          19.53 GiB    33.30 GiB   +70.5%
3B          37.54 GiB    54.55 GiB   +45.3%
7B          too large         skip        -
(llm-systems-tutorial) abhishek.maiti@dgx-97:~/projects/llm-systems-tutorial/chapters/02-memory-compute-budget$ python measure_memory.py
/home/abhishek.maiti/projects/llm-systems-tutorial/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
preset      predicted     measured    delta
--------------------------------------------
tiny       816.47 MiB     6.03 GiB  +656.0%
small        3.25 GiB    10.23 GiB  +215.2%
medium       9.02 GiB    19.67 GiB  +118.2%
1B          23.28 GiB    33.30 GiB   +43.0%
3B          43.79 GiB    54.55 GiB   +24.6%
7B          too large         skip        -
(llm-systems-tutorial) abhishek.maiti@dgx-97:~/projects/llm-systems-tutorial/chapters/02-memory-compute-budget$ python measure_memory.py
/home/abhishek.maiti/projects/llm-systems-tutorial/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
preset      predicted     measured    delta
--------------------------------------------
tiny         1.04 GiB     6.03 GiB  +477.7%
small        4.23 GiB    10.23 GiB  +141.8%
medium      11.64 GiB    19.67 GiB   +69.0%
1B          28.53 GiB    33.30 GiB   +16.7%
3B          52.54 GiB    54.55 GiB    +3.8%
7B          too large         skip        -