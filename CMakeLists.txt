#
# Copyright 2023-2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.16)
project(tflite_neutron_delegate)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)
option(BUILD_FOR_LITERT "Configure neutron delegate build for LiteRT" OFF)

# Build in debug mode by default
if (NOT CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_BUILD_TYPE Debug CACHE INTERNAL "")
endif()

set(DELEGATE_SRC_DIR        ${CMAKE_CURRENT_SOURCE_DIR})

if(${BUILD_FOR_LITERT})
  set(TFLITE_GIT_REPOSITORY "https://github.com/nxp-imx/litert-imx")
  set(TFLITE_GIT_TAG "lf-6.12.34_2.1.0")
  set(NEUTRON_DELEGATE_TARGET "litert_neutron_delegate")
  set(TFLITE_SUB_PATH "tflite")
  list(APPEND NEUTRON_DELEGATE_OPTIONS "-DBUILD_FOR_LITERT")
else()
  set(TFLITE_GIT_REPOSITORY "https://github.com/nxp-imx/tensorflow-imx")
  set(TFLITE_GIT_TAG "lf-6.12.34_2.1.0")
  set(NEUTRON_DELEGATE_TARGET "neutron_delegate")
  set(TFLITE_SUB_PATH "tensorflow/lite")
endif()
find_package(tensorflow REQUIRED)
list(APPEND NEUTRON_DELEGATE_SRCS
  ${tensorflow_SOURCE_DIR}/${TFLITE_SUB_PATH}/delegates/utils/simple_delegate.cc
  ${DELEGATE_SRC_DIR}/neutron_delegate.cc
  ${DELEGATE_SRC_DIR}/neutron_delegate_adaptor.cc
  ${DELEGATE_SRC_DIR}/neutron_delegate_dmabuf.cc
)

execute_process(
    COMMAND git rev-parse --short=8 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(GIT_COMMIT_HASH)
    message(STATUS "Current git commit hash (short): ${GIT_COMMIT_HASH}")
else()
    message(WARNING "Failed to retrieve git commit hash. Not a Git repository?")
    set(GIT_COMMIT_HASH "unknown")
endif()
add_definitions(-DGIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

add_library(${NEUTRON_DELEGATE_TARGET} SHARED ${NEUTRON_DELEGATE_SRCS})
target_include_directories(${NEUTRON_DELEGATE_TARGET}
    PUBLIC
    $<TARGET_PROPERTY:TensorFlow::tensorflow-lite,INCLUDE_DIRECTORIES>
)

if(NOT DEFINED NEUTRON_DRIVER_PATH)
  list(APPEND NEUTRON_DELEGATE_DEPENDENCIES NeutronDriver)
else()
  if(NOT DEFINED ENV{SDKTARGETSYSROOT})
    target_link_options(${NEUTRON_DELEGATE_TARGET} PRIVATE "-Wl,-z,max-page-size=16384")
  endif()
  include_directories(${NEUTRON_DRIVER_PATH}/include)
  list(APPEND NEUTRON_DELEGATE_DEPENDENCIES ${NEUTRON_DRIVER_PATH}/library/libNeutronDriver.so)
endif()

target_compile_options(${NEUTRON_DELEGATE_TARGET} PUBLIC ${NEUTRON_DELEGATE_OPTIONS})
target_link_libraries(${NEUTRON_DELEGATE_TARGET} PUBLIC ${NEUTRON_DELEGATE_DEPENDENCIES})
