// models/index.json is what a packaged application reads before it loads
// anything, so it has to fail loudly rather than creatively: a default naming
// no bundled model, a path escaping the package, or a bad digest must all be
// refused before a single weight is touched.
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

#include "check.hpp"
#include "diamond_model/model_index.hpp"

namespace {

const char* kValid = R"({
  "index_version": 1,
  "defaults": { "soo": "soo/2.0.0" },
  "models": [
    {
      "family": "soo",
      "version": "2.0.0",
      "path": "soo/2.0.0",
      "model_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "runtime_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    {
      "family": "min",
      "version": "0.1.0",
      "path": "min/0.1.0",
      "model_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "runtime_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    }
  ]
})";

std::filesystem::path write_index(const std::filesystem::path& directory,
                                  const std::string& contents) {
    std::filesystem::create_directories(directory);
    std::ofstream file(directory / "index.json", std::ios::binary | std::ios::trunc);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return directory;
}

bool rejects(const std::filesystem::path& directory) {
    try {
        (void)diamond_model::load_model_index(directory);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

std::string with(const std::string& before, const std::string& after) {
    std::string text = kValid;
    const size_t position = text.find(before);
    if (position == std::string::npos) return text;
    return text.replace(position, before.size(), after);
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 2, "usage: model_index_test <scratch-dir>");
    const std::filesystem::path scratch(argv[1]);
    std::error_code ignored;
    std::filesystem::remove_all(scratch, ignored);

    const auto valid = write_index(scratch / "valid", kValid);
    const auto index = diamond_model::load_model_index(valid);
    CHECK_EQ(index.models.size(), static_cast<std::size_t>(2));

    const auto* soo = index.default_for("soo");
    REQUIRE(soo != nullptr, "the soo default should resolve");
    CHECK_EQ(soo->version, std::string("2.0.0"));
    CHECK_EQ(soo->root, valid / "soo" / "2.0.0");

    // A family with no declared default is not an error: a release may bundle
    // one model and still list others.
    CHECK(index.default_for("min") == nullptr);
    CHECK(index.default_for("nothing-like-this") == nullptr);

    CHECK(rejects(write_index(scratch / "missing-model",
                              with("\"soo\": \"soo/2.0.0\"", "\"soo\": \"soo/9.9.9\""))));
    CHECK(rejects(write_index(scratch / "escape",
                              with("\"path\": \"soo/2.0.0\"", "\"path\": \"../../etc\""))));
    CHECK(rejects(write_index(
        scratch / "digest",
        with("\"model_sha256\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
             "\"model_sha256\": \"nope\""))));
    CHECK(rejects(write_index(scratch / "version",
                              with("\"index_version\": 1", "\"index_version\": 2"))));
    CHECK(rejects(scratch / "not-written-at-all"));

    std::filesystem::remove_all(scratch, ignored);
    return soo_test::report("model_index_test");
}
