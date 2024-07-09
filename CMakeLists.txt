#
# Copyright 2023-2024 NXP
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

# Build in debug mode by default
if (NOT CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_BUILD_TYPE Debug CACHE INTERNAL "")
endif()

set(DELEGATE_SRC_DIR        ${CMAKE_CURRENT_SOURCE_DIR})


find_package(tensorflow REQUIRED)
list(APPEND NEUTRON_DELEGATE_SRCS
  ${tensorflow_SOURCE_DIR}/tensorflow/lite/delegates/utils/simple_delegate.cc
  ${DELEGATE_SRC_DIR}/neutron_delegate.cc
  ${DELEGATE_SRC_DIR}/neutron_delegate_adaptor.cc
  ${DELEGATE_SRC_DIR}/neutron_delegate_utils.cc
)

include_directories(${CMAKE_BINARY_DIR}/flatbuffers/include)
include_directories("${tensorflow_SOURCE_DIR}")
include_directories("${DELEGATE_SRC_DIR}/include")
link_directories(${DELEGATE_SRC_DIR})

add_library(neutron_delegate SHARED ${NEUTRON_DELEGATE_SRCS})

if(NOT DEFINED NEUTRON_DRIVER_PATH)
  list(APPEND NEUTRON_DELEGATE_DEPENDENCIES NeutronConverter NeutronDriver)
else()
  include_directories(${NEUTRON_DRIVER_PATH}/include)
  list(APPEND NEUTRON_DELEGATE_DEPENDENCIES ${NEUTRON_DRIVER_PATH}/library/libNeutronDriver.so)
  list(APPEND NEUTRON_DELEGATE_DEPENDENCIES ${NEUTRON_DRIVER_PATH}/library/libNeutronConverter.so)
endif()

target_compile_options(neutron_delegate PUBLIC ${NEUTRON_DELEGATE_OPTIONS})
target_link_libraries(neutron_delegate PUBLIC ${NEUTRON_DELEGATE_DEPENDENCIES})
