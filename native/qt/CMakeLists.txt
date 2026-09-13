# 6.6 is a hard floor, not a preference: the QML uses Dial.startAngle and
# Dial.endAngle, which do not exist before it. A too-old Qt otherwise builds
# cleanly and fails when the application loads its interface.
find_package(Qt6 6.6 REQUIRED COMPONENTS Core Gui Network Qml Quick QuickControls2 Multimedia)

option(DIAMOND_QML_PREVIEW "Enable Qt 6.12 QML hot reload for development" OFF)
if(DIAMOND_QML_PREVIEW AND Qt6_VERSION VERSION_LESS 6.12)
    message(FATAL_ERROR "DIAMOND_QML_PREVIEW requires Qt 6.12 or newer (use the isolated preview kit)")
endif()

if(DIAMOND_BUILD_QT_SOO)
    find_package(Torch REQUIRED)
    if(NOT TARGET alphadiamond::diamond_model)
        message(FATAL_ERROR "DIAMOND_BUILD_QT_SOO requires DIAMOND_BUILD_LIBTORCH_PROBE=ON")
    endif()
endif()

add_library(diamond_qt_backend STATIC
    ai_worker.cpp
    ai_worker.hpp
    search_telemetry.cpp
    search_telemetry.hpp
    model_catalog.cpp
    model_catalog.hpp
    soo_search_runtime.cpp
    soo_search_runtime.hpp
    native_controller.cpp
    native_controller.hpp
    native_move_player.cpp
    native_move_player.hpp
)
target_link_libraries(diamond_qt_backend PUBLIC
    Qt6::Core Qt6::Gui Qt6::Network Qt6::Qml Qt6::Quick Qt6::Multimedia
    alphadiamond::soo_core alphadiamond::soo_search
)
target_include_directories(diamond_qt_backend PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_features(diamond_qt_backend PUBLIC cxx_std_20)
set_target_properties(diamond_qt_backend PROPERTIES AUTOMOC ON)
if(TARGET alphadiamond::diamond_orchestration)
    target_sources(diamond_qt_backend PRIVATE
        rating_bridge.cpp
        rating_bridge.hpp
    )
    target_link_libraries(diamond_qt_backend PUBLIC
        alphadiamond::diamond_orchestration)
    target_compile_definitions(diamond_qt_backend PRIVATE DIAMOND_QT_HAS_RATINGS)
endif()
if(DIAMOND_BUILD_QT_SOO)
    target_link_libraries(diamond_qt_backend PUBLIC alphadiamond::diamond_model ${TORCH_LIBRARIES})
    target_include_directories(diamond_qt_backend PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(diamond_qt_backend
        PUBLIC DIAMOND_QT_HAS_SOO
        PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE QT_NO_KEYWORDS)
endif()
if(MSVC)
    target_compile_options(diamond_qt_backend PRIVATE /W4)
endif()

qt_add_executable(diamond_qt
    main.cpp
    native_chrome.cpp
    native_chrome.hpp
)

set(QML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/qml")
set(ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/assets")

qt_add_resources(diamond_qt qml_resources
    PREFIX "/qml"
    BASE "${QML_ROOT}"
    FILES
        "${QML_ROOT}/Main.qml"
        "${QML_ROOT}/ActionButton.qml"
        "${QML_ROOT}/AppDialog.qml"
        "${QML_ROOT}/Board.qml"
        "${QML_ROOT}/BoardRotationControl.qml"
        "${QML_ROOT}/GamePanel.qml"
        "${QML_ROOT}/AiPanel.qml"
        "${QML_ROOT}/TelemetryChart.qml"
        "${QML_ROOT}/PositionOutlookPanel.qml"
        "${QML_ROOT}/DecisionValuePanel.qml"
        "${QML_ROOT}/MovePreferencePanel.qml"
        "${QML_ROOT}/ModelsDialog.qml"
        "${QML_ROOT}/SearchComputePanel.qml"
        "${QML_ROOT}/HistoryDrawer.qml"
        "${QML_ROOT}/Hole.qml"
        "${QML_ROOT}/HistoryPanel.qml"
        "${QML_ROOT}/ReplayBar.qml"
        "${QML_ROOT}/Icon.qml"
        "${QML_ROOT}/NewMatchDialog.qml"
        "${QML_ROOT}/PanelScrollBar.qml"
        "${QML_ROOT}/PanelSection.qml"
        "${QML_ROOT}/Piece.qml"
        "${QML_ROOT}/ReorderButton.qml"
        "${QML_ROOT}/ResultDialog.qml"
        "${QML_ROOT}/SegmentedControl.qml"
        "${QML_ROOT}/SidePanel.qml"
        "${QML_ROOT}/SoundDialog.qml"
        "${QML_ROOT}/TitleBar.qml"
        "${QML_ROOT}/TitleMenu.qml"
        "${QML_ROOT}/WindowButton.qml"
        "${QML_ROOT}/Style/Theme.qml"
        "${QML_ROOT}/Style/qmldir"
)

qt_add_resources(diamond_qt font_resources
    PREFIX "/fonts"
    BASE "${ASSET_ROOT}/fonts"
    FILES
        "${ASSET_ROOT}/fonts/GoogleSansFlex-Regular.ttf"
        "${ASSET_ROOT}/fonts/GoogleSansFlex-Medium.ttf"
        "${ASSET_ROOT}/fonts/GoogleSansFlex-Bold.ttf"
)

qt_add_resources(diamond_qt model_source_icons
    PREFIX "/assets/models"
    BASE "${ASSET_ROOT}"
    FILES
        "${ASSET_ROOT}/github.svg"
        "${ASSET_ROOT}/huggingface.svg"
)

target_link_libraries(diamond_qt PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Multimedia
    diamond_qt_backend
)
target_compile_features(diamond_qt PRIVATE cxx_std_20)
set_target_properties(diamond_qt PROPERTIES AUTOMOC ON WIN32_EXECUTABLE TRUE)

# This app uses explicit resources rather than qt_add_qml_module. Supply their
# maps to qmlpreview so edits to source files update the embedded qrc: URLs.
set_target_properties(diamond_qt PROPERTIES QT_QML_NO_QML_PREVIEW TRUE)
if(DIAMOND_QML_PREVIEW)
    find_program(DIAMOND_QML_PREVIEW_EXECUTABLE NAMES qmlpreview qmlpreview.exe
        HINTS "${Qt6_DIR}/../../../bin" "${Qt6_DIR}/../../../lib/qt6/bin"
        NO_DEFAULT_PATH REQUIRED)
    target_compile_definitions(diamond_qt PRIVATE QT_QML_DEBUG)
    get_target_property(preview_resources diamond_qt _qt_generated_qrc_files)
    set(preview_arguments)
    foreach(resource IN LISTS preview_resources)
        list(APPEND preview_arguments --resource "${resource}")
    endforeach()
    add_custom_target(diamond_qt_preview
        COMMAND "${DIAMOND_QML_PREVIEW_EXECUTABLE}" --verbose ${preview_arguments} $<TARGET_FILE:diamond_qt>
        DEPENDS diamond_qt
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        USES_TERMINAL VERBATIM)
endif()

if(MSVC)
    target_compile_options(diamond_qt PRIVATE /W4)
endif()
if(WIN32)
    target_link_libraries(diamond_qt PRIVATE dwmapi shell32 user32)
endif()

if(BUILD_TESTING)
    if(DIAMOND_QML_PREVIEW)
        add_executable(diamond_qt_preview_test tests/qml_preview_test.cpp)
        target_link_libraries(diamond_qt_preview_test PRIVATE Qt6::Core)
        add_test(NAME diamond_qt_preview_test COMMAND diamond_qt_preview_test
            "${DIAMOND_QML_PREVIEW_EXECUTABLE}" $<TARGET_FILE:diamond_qt>
            "${CMAKE_CURRENT_BINARY_DIR}/.qt/rcc/qml_resources.qrc")
    endif()
    if(WIN32)
        add_executable(diamond_qt_native_chrome_test
            tests/native_chrome_test.cpp native_chrome.cpp native_chrome.hpp)
        target_link_libraries(diamond_qt_native_chrome_test PRIVATE
            Qt6::Core Qt6::Gui dwmapi shell32 user32)
        target_include_directories(diamond_qt_native_chrome_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        set_target_properties(diamond_qt_native_chrome_test PROPERTIES AUTOMOC ON)
        add_test(NAME diamond_qt_native_chrome_test COMMAND diamond_qt_native_chrome_test)
        add_test(NAME diamond_qt_window_smoke COMMAND diamond_qt --window-smoke)
    endif()
    if(DIAMOND_BUILD_QT_SOO)
        add_executable(diamond_qt_min_position_audit min_position_audit.cpp)
        target_link_libraries(diamond_qt_min_position_audit PRIVATE diamond_qt_backend)
        target_compile_definitions(diamond_qt_min_position_audit PRIVATE QT_NO_KEYWORDS)
        add_executable(diamond_qt_min_runtime_test tests/min_runtime_test.cpp)
        target_link_libraries(diamond_qt_min_runtime_test PRIVATE diamond_qt_backend)
        target_compile_definitions(diamond_qt_min_runtime_test PRIVATE QT_NO_KEYWORDS)
        set(DIAMOND_QT_MIN_TEST_ARTIFACT "" CACHE PATH "Min deployment artifact for Qt inference test")
        if(DIAMOND_QT_MIN_TEST_ARTIFACT)
            add_test(NAME diamond_qt_min_runtime_test COMMAND diamond_qt_min_runtime_test
                "${DIAMOND_QT_MIN_TEST_ARTIFACT}")
            set_tests_properties(diamond_qt_min_runtime_test PROPERTIES
                ENVIRONMENT "DIAMOND_TORCH_THREADS=2")
        endif()
    endif()
    add_executable(diamond_qt_controller_contract
        tests/controller_contract.cpp
    )
    target_link_libraries(diamond_qt_controller_contract PRIVATE
        diamond_qt_backend
    )
    target_compile_features(diamond_qt_controller_contract PRIVATE cxx_std_20)
    set_target_properties(diamond_qt_controller_contract PROPERTIES AUTOMOC ON)
    if(MSVC)
        target_compile_options(diamond_qt_controller_contract PRIVATE /W4)
    endif()
    add_test(NAME diamond_qt_controller_contract COMMAND diamond_qt_controller_contract)
    add_test(NAME diamond_qt_replay_smoke COMMAND diamond_qt --replay-smoke)
    add_test(NAME diamond_qt_analysis_smoke COMMAND diamond_qt --analysis-smoke)

    add_executable(diamond_qt_model_catalog_test tests/model_catalog_test.cpp)
    target_link_libraries(diamond_qt_model_catalog_test PRIVATE diamond_qt_backend)
    target_compile_features(diamond_qt_model_catalog_test PRIVATE cxx_std_20)
    set_target_properties(diamond_qt_model_catalog_test PROPERTIES AUTOMOC ON)
    if(MSVC)
        target_compile_options(diamond_qt_model_catalog_test PRIVATE /W4)
    endif()
    add_test(NAME diamond_qt_model_catalog_test COMMAND diamond_qt_model_catalog_test)

    if(TARGET alphadiamond::diamond_orchestration)
        add_executable(diamond_qt_rating_bridge_test tests/rating_bridge_test.cpp)
        target_link_libraries(diamond_qt_rating_bridge_test PRIVATE diamond_qt_backend)
        target_compile_features(diamond_qt_rating_bridge_test PRIVATE cxx_std_20)
        set_target_properties(diamond_qt_rating_bridge_test PROPERTIES AUTOMOC ON)
        if(MSVC)
            target_compile_options(diamond_qt_rating_bridge_test PRIVATE /W4)
        endif()
        add_test(NAME diamond_qt_rating_bridge_test COMMAND diamond_qt_rating_bridge_test)
    endif()

    get_filename_component(DIAMOND_QT_ENV_PREFIX "${Qt6_DIR}/../../.." ABSOLUTE)
    if(EXISTS "${DIAMOND_QT_ENV_PREFIX}/plugins")
        set(DIAMOND_QT_PLUGIN_DIR "${DIAMOND_QT_ENV_PREFIX}/plugins")
        set(DIAMOND_QT_QML_DIR "${DIAMOND_QT_ENV_PREFIX}/qml")
    else()
        set(DIAMOND_QT_PLUGIN_DIR "${DIAMOND_QT_ENV_PREFIX}/lib/qt6/plugins")
        set(DIAMOND_QT_QML_DIR "${DIAMOND_QT_ENV_PREFIX}/lib/qt6/qml")
    endif()
    set(DIAMOND_QT_RUNTIME_PATH_MODIFICATIONS
        "PATH=path_list_prepend:${DIAMOND_QT_ENV_PREFIX}/bin"
        "PATH=path_list_prepend:${DIAMOND_QT_ENV_PREFIX}/lib/qt6/bin")
    if(DIAMOND_BUILD_QT_SOO)
        get_filename_component(DIAMOND_TORCH_PACKAGE_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
        list(APPEND DIAMOND_QT_RUNTIME_PATH_MODIFICATIONS
            "PATH=path_list_prepend:${DIAMOND_TORCH_PACKAGE_ROOT}/lib")
    endif()
    set(DIAMOND_QT_TESTS
        diamond_qt_controller_contract
        diamond_qt_analysis_smoke
        diamond_qt_replay_smoke
        diamond_qt_model_catalog_test)
    if(TARGET diamond_qt_rating_bridge_test)
        list(APPEND DIAMOND_QT_TESTS diamond_qt_rating_bridge_test)
    endif()
    set_tests_properties(${DIAMOND_QT_TESTS} PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_PLUGIN_PATH=${DIAMOND_QT_PLUGIN_DIR};QML_IMPORT_PATH=${DIAMOND_QT_QML_DIR};QT_FORCE_STDERR_LOGGING=1"
        ENVIRONMENT_MODIFICATION "${DIAMOND_QT_RUNTIME_PATH_MODIFICATIONS}"
    )
    if(WIN32)
        # Offscreen tests cannot exercise Windows caption messages.
        set_tests_properties(diamond_qt_native_chrome_test diamond_qt_window_smoke PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            ENVIRONMENT "QT_QPA_PLATFORM=windows;QT_PLUGIN_PATH=${DIAMOND_QT_PLUGIN_DIR};QML_IMPORT_PATH=${DIAMOND_QT_QML_DIR};QT_FORCE_STDERR_LOGGING=1"
            ENVIRONMENT_MODIFICATION "${DIAMOND_QT_RUNTIME_PATH_MODIFICATIONS}"
            TIMEOUT 30)
    endif()
    if(DIAMOND_QML_PREVIEW)
        set_tests_properties(diamond_qt_preview_test PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            ENVIRONMENT "QT_QPA_PLATFORM=windows;QT_PLUGIN_PATH=${DIAMOND_QT_PLUGIN_DIR};QML_IMPORT_PATH=${DIAMOND_QT_QML_DIR};QMLPREVIEW_HOTRELOAD=1;QT_FORCE_STDERR_LOGGING=1"
            ENVIRONMENT_MODIFICATION "${DIAMOND_QT_RUNTIME_PATH_MODIFICATIONS}"
            TIMEOUT 50)
    endif()
endif()

set(MOVE_SOUND "${ASSET_ROOT}/sounds/move.wav")
foreach(sound_target diamond_qt diamond_qt_controller_contract)
    if(TARGET ${sound_target})
        add_custom_command(TARGET ${sound_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${sound_target}>/assets/sounds"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MOVE_SOUND}" "$<TARGET_FILE_DIR:${sound_target}>/assets/sounds/move.wav"
            VERBATIM)
    endif()
endforeach()

install(TARGETS diamond_qt RUNTIME DESTINATION bin BUNDLE DESTINATION .)
install(FILES "${MOVE_SOUND}" DESTINATION bin/assets/sounds)
if(TARGET alphadiamond-rating-sync)
    install(PROGRAMS "${PROJECT_SOURCE_DIR}/tools/sync_ratings.sh" DESTINATION bin)
endif()
