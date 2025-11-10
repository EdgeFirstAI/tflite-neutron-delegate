#
# Copyright 2022-2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

if (NOT NEUTRON_INTEGRATION)
  FetchContent_Declare(
    tensorflow
    GIT_REPOSITORY ${TFLITE_GIT_REPOSITORY}
    GIT_TAG ${TFLITE_GIT_TAG}
    GIT_SHALLOW    TRUE
  )
else()
  message(STATUS "Using LOCAL Tensorflow")
  FetchContent_Declare(
    tensorflow
    SOURCE_DIR "${DELEGATE_SRC_DIR}/../tflite"
  )
  FetchContent_MakeAvailable(tensorflow)
endif()

FetchContent_GetProperties(tensorflow)
if(NOT tensorflow_POPULATED)
  FetchContent_Populate(tensorflow)
endif()

set(TFLITE_BUILD_SHARED_LIB ON CACHE BOOL "Build shared library instead of static" FORCE)
add_subdirectory("${tensorflow_SOURCE_DIR}/${TFLITE_SUB_PATH}"
                 "${tensorflow_BINARY_DIR}")
get_target_property(TFLITE_SOURCE_DIR tensorflow-lite SOURCE_DIR)

if(NOT TFLITE_LIB_LOC OR NOT EXISTS ${TFLITE_LIB_LOC})
  add_library(TensorFlow::tensorflow-lite ALIAS tensorflow-lite)
else()
  add_library(TensorFlow::tensorflow-lite UNKNOWN IMPORTED)
  set_target_properties(TensorFlow::tensorflow-lite PROPERTIES
    IMPORTED_LOCATION ${TFLITE_LIB_LOC}
    INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:tensorflow-lite,INTERFACE_INCLUDE_DIRECTORIES>
  )
  set_target_properties(tensorflow-lite PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

list(APPEND NEUTRON_DELEGATE_DEPENDENCIES TensorFlow::tensorflow-lite)
list(APPEND NEUTRON_DELEGATE_SRCS ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc)
