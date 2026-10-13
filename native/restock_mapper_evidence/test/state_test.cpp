#include "restock_mapper_evidence/state.hpp"
#include <cassert>
#include <functional>
using namespace restock_mapper_evidence;
const Identity id{"session", "scene", "cameras", "localization"};
State fresh() { return State("native-nonce", id, {{"front", "front_optical"}, {"rear", "rear_optical"}}); }
void rejects(const std::function<void()>& fn) {
  bool rejected=false;
  try { fn(); } catch (const std::runtime_error&) { rejected=true; }
  assert(rejected);
}
void open(State& s) {
  s.certify("challenge", id, {0,0,0}, true, true);
  s.admit("native-nonce", "challenge", id, {0,0,0}, true, true, 100, 1000);
}
int main() {
  { auto s=fresh(); rejects([&]{ s.certify("c",id,{0,1},true,true); }); assert(s.phase()==State::Phase::Fault); }
  { auto s=fresh(); rejects([&]{ s.certify("c",id,{0},false,true); }); }
  { auto s=fresh(); rejects([&]{ s.certify("c",id,{},true,true); }); }
  { auto s=fresh(); s.certify("c",id,{0},true,true); rejects([&]{s.admit("wrong","c",id,{0},true,true,1,1);}); }
  { auto s=fresh(); open(s); rejects([&]{s.match("front_optical",101,101,1001);}); }
  Acquisition a{id,"front","front_optical",1,101,1001};
  { auto s=fresh(); open(s); auto same=a; same.steady=1000;
    rejects([&]{s.register_acquisition(s.nonce(),same,101,1001);}); }
  { auto s=fresh(); open(s); auto same=a; same.stamp=100;
    rejects([&]{s.register_acquisition(s.nonce(),same,101,1001);}); }
  { auto s=fresh(); s.certify("c",id,{0},true,true);
    rejects([&]{s.admit(s.nonce(),"c",id,{1},true,true,100,1000);}); }
  { auto s=fresh(); open(s); auto wrong=a; wrong.identity[1]="changed_scene";
    rejects([&]{s.register_acquisition(s.nonce(),wrong,101,1001);}); }
  { auto s=fresh(); open(s); s.register_acquisition(s.nonce(),a,101,1001);
    auto actual=s.match(a.frame,a.stamp,102,1002); s.finish(actual,false);
    assert(s.ledger().empty()); rejects([&]{s.snapshot(102,1002);}); }
  { auto s=fresh(); open(s); s.register_acquisition(s.nonce(),a,101,1001);
    rejects([&]{s.register_acquisition(s.nonce(),a,101,1001);}); }
  { auto s=fresh(); open(s); s.register_acquisition(s.nonce(),a,101,1001);
    s.finish(s.match(a.frame,a.stamp,102,1002),true);
    rejects([&]{s.snapshot(102,1002);});
    Acquisition b{id,"rear","rear_optical",1,103,1003};
    s.register_acquisition(s.nonce(),b,103,1003); s.finish(s.match(b.frame,b.stamp,104,1004),true);
    assert(s.snapshot(104,1004)==1 && s.revision()==2);
    rejects([&]{s.snapshot(400000104,400001004);}); }
  { auto s=fresh(); open(s); rejects([&]{s.clocks(99,1001);}); }
  { auto s=fresh(); open(s); rejects([&]{s.clocks(600000000101,600000001001);}); }
}
