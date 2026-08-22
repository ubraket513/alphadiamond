#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "alphadiamond::soo_core" for configuration "Debug"
set_property(TARGET alphadiamond::soo_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(alphadiamond::soo_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/soo_core.lib"
  )

list(APPEND _cmake_import_check_targets alphadiamond::soo_core )
list(APPEND _cmake_import_check_files_for_alphadiamond::soo_core "${_IMPORT_PREFIX}/lib/soo_core.lib" )

# Import target "alphadiamond::soo_search" for configuration "Debug"
set_property(TARGET alphadiamond::soo_search APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(alphadiamond::soo_search PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/soo_search.lib"
  )

list(APPEND _cmake_import_check_targets alphadiamond::soo_search )
list(APPEND _cmake_import_check_files_for_alphadiamond::soo_search "${_IMPORT_PREFIX}/lib/soo_search.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
