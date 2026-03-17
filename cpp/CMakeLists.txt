cmake_minimum_required(VERSION 3.18)
project(krakencpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(krakencpp
    src/bindings.cpp
    src/kraken_ws.cpp
    src/execution_engine.cpp
)

target_include_directories(krakencpp PRIVATE include)

pybind11_add_module(cuda_features
    src/cuda_features_bindings.cpp
    src/cuda_features.cu
)

target_include_directories(cuda_features PRIVATE include)
set_target_properties(cuda_features PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(cuda_correlation
    src/cuda_correlation_bindings.cpp
    src/cuda_correlation.cu
)

target_include_directories(cuda_correlation PRIVATE include)
set_target_properties(cuda_correlation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(cuda_rsi
    src/cuda_rsi_bindings.cpp
    src/cuda_rsi.cu
)

target_include_directories(cuda_rsi PRIVATE include)
set_target_properties(cuda_rsi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(cuda_atr
    src/cuda_atr_bindings.cpp
    src/cuda_atr.cu
)

target_include_directories(cuda_atr PRIVATE include)
set_target_properties(cuda_atr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(cuda_bollinger
    src/cuda_bollinger_bindings.cpp
    src/cuda_bollinger.cu
)

target_include_directories(cuda_bollinger PRIVATE include)
set_target_properties(cuda_bollinger PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(cuda_northstar
    src/cuda_northstar_bindings.cpp
    src/cuda_northstar.cu
)

target_include_directories(cuda_northstar PRIVATE include)
set_target_properties(cuda_northstar PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS krakencpp
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)

install(TARGETS cuda_features
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)

install(TARGETS cuda_correlation
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)

install(TARGETS cuda_rsi
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)

install(TARGETS cuda_atr
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)

install(TARGETS cuda_bollinger
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)

install(TARGETS cuda_northstar
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
    ARCHIVE DESTINATION ${Python3_SITEARCH}
)
