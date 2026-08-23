// The physical action id is source * 73 + destination, and it round-trips for
// every pair on the board -- not only the four the old smoke test sampled.
#include "check.hpp"
#include "soo/action.hpp"
#include "soo/board.hpp"

int main() {
    for (int source = 0; source < soo::kBoardSize; ++source) {
        for (int destination = 0; destination < soo::kBoardSize; ++destination) {
            const auto action = soo::encode_action(source, destination);
            CHECK(action >= 0 && action < soo::kActionSize);
            int decoded_source = -1;
            int decoded_destination = -1;
            soo::decode_action(action, decoded_source, decoded_destination);
            CHECK_EQ(decoded_source, source);
            CHECK_EQ(decoded_destination, destination);
        }
    }

    // Out of range ids must throw rather than silently decode.
    bool threw = false;
    try {
        int source = 0;
        int destination = 0;
        soo::decode_action(soo::kActionSize, source, destination);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        int source = 0;
        int destination = 0;
        soo::decode_action(-1, source, destination);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);

    return soo_test::report("action_codec_test");
}
