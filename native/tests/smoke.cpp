#include <cassert>

#include "soo/action.hpp"
#include "soo/board.hpp"

int main() {
    for (int source : {0, 1, 36, soo::kBoardSize - 1}) {
        for (int destination : {0, 2, 41, soo::kBoardSize - 1}) {
            const auto action = soo::encode_action(source, destination);
            int decoded_source = -1;
            int decoded_destination = -1;
            soo::decode_action(action, decoded_source, decoded_destination);
            assert(decoded_source == source);
            assert(decoded_destination == destination);
        }
    }

    return 0;
}
