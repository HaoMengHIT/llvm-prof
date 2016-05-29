/**
 *
 *How does TimingSource work?
 *1. Initialization Stage
 *
 *      Check out TimingSource.cpp
 *
 *2. Parse -timing option and create corrending objects
 *
 *      Check out llvm-prof.cpp
 *
 *3. Calculate MPI time
 *
 *      At passes.cpp
 *      if(isa<MPITiming>(S) && MpiTiming < DBL_EPSILON)//Only enter this if statement once
 *      {
 *          auto MT = cast<MPITiming>(S);
 *          auto S = PI.getAllTrapedValues(MPIFullInfo);//this S is not the same as the above S
 *          ...
 *          for(auto I : S)//for each MPI instruction I, get I's time
 *          {
 *              ...
 *  ------------double timing = MT->count(*I, PI.getExecutionCount(BB), PI.getExecutionCount(CI));
 *  |           ...
 *  |           MpiTiming += timing;
 *  |           }
 *  |       }
 *  |   }
 *  |
 *  |
 *  |
 *  |   At TimingSource.cpp
 *  --->LatencyTiming::count(const llvm::Instruction &I,double bfreq,double total)
 *      {
 *          //R is MPI_SIZE
 *          first, determin the type of I(the variable C)-----------------------------------enum MPICategoryType
 *          if I is p2p operation,        use bfreq*latency+total/bandwith                  {
 *          if I is collective operation, use bfreq*latency+C*total*log2(R)/bandwith            MPI_CT_P2P     =0,
 *          else                          use 2*R*(bfreq*latency+total/bandwith)                MPI_CT_REDUCE  =1,
 *      }                                                                                       MPI_CT_REDUCE2 =2,
 *                                                                                              MPI_CT_NSIDES  =3,
 *                                                                                          }
 */
#include "passes.h"
#include <ProfileInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <fstream>
#include <iterator>
#include <float.h>
#include "ValueUtils.h"

using namespace llvm;

namespace {
#ifndef NDEBUG
   cl::opt<bool> TimingDebug("timing-debug", cl::desc("print more detail for timing mode"));
#endif
   cl::opt<std::string> TimingIgnore("timing-ignore",
                                     cl::desc("ignore list for timing mode"),
                                     cl::init(""));
};

char ProfileInfoConverter::ID = 0;
void ProfileInfoConverter::getAnalysisUsage(AnalysisUsage &AU) const
{
   AU.addRequired<ProfileInfo>();
   AU.setPreservesAll();
}

bool ProfileInfoConverter::runOnModule(Module &M)
{
   ProfileInfo& PI = getAnalysis<ProfileInfo>();

   std::vector<unsigned> Counters;
   for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB){
         Counters.push_back(PI.getExecutionCount(BB));
      }
   }

   Writer.write(BlockInfo, Counters);

   return false;
}


bool ProfileInfoCompare::run()
{
#define CRITICAL_EQUAL(what) if(Lhs.get##what() != Rhs.get##what()){\
      errs()<<#what" differ\n";\
      return 0;\
   }
#define WARN_EQUAL(what) if(Lhs.getRaw##what() != Rhs.getRaw##what()){\
      errs()<<#what" differ\n";\
   }

   CRITICAL_EQUAL(NumExecutions);
   WARN_EQUAL(BlockCounts);
   WARN_EQUAL(EdgeCounts);
   WARN_EQUAL(FunctionCounts);
   WARN_EQUAL(ValueCounts);
   WARN_EQUAL(SLGCounts);
   if(Lhs.getRawValueCounts().size() == Rhs.getRawValueCounts().size()){
      for(uint i=0;i<Lhs.getRawValueCounts().size();++i){
         if(Lhs.getRawValueContent(i) != Rhs.getRawValueContent(i))
            errs()<<"ValueContent at "<<i<<" differ\n";
      }
   }
   return 0;
#undef CRITICAL_EQUAL
}

char ProfileInfoComm::ID = 0;
void ProfileInfoComm::getAnalysisUsage(AnalysisUsage &AU) const
{
   AU.setPreservesAll();
   AU.addRequired<ProfileInfo>();
}
bool ProfileInfoComm::runOnModule(Module &M)
{
   ProfileInfo& PI = getAnalysis<ProfileInfo>();
   auto S = PI.getAllTrapedValues(MPIFullInfo);
   auto U = PI.getAllTrapedValues(MPInfo);
   if(U.size()>0) outs()<<"Notice: Old Mpi Profiling Format\n";
   S.insert(S.end(), U.begin(), U.end());
   for(auto I : S){
      const CallInst* CI = cast<CallInst>(I);
      const BasicBlock* BB = CI->getParent();
      size_t BFreq = PI.getExecutionCount(BB);
      double MpiComm = PI.getExecutionCount(CI);//LTR->Comm_amount(*I,BFreq,PI.getExecutionCount(CI));
      if(CI == NULL) continue;
      Value* CV = const_cast<CallInst*>(CI)->getCalledValue();
      Function* func = dyn_cast<Function>(lle::castoff(CV));
      if(func == NULL)
         errs()<<"No func!\n";
      StringRef str = func->getName();
      if(str.startswith("mpi_")){
         outs()<<str<<"\t"<<(size_t)(MpiComm/BFreq)<<"\t" << MpiComm<<"\t"<<BFreq<<"\n";
      }
   }
}


char ProfileTimingPrint::ID = 0;
void ProfileTimingPrint::getAnalysisUsage(AnalysisUsage &AU) const
{
   AU.setPreservesAll();
   AU.addRequired<ProfileInfo>();
}

bool ProfileTimingPrint::runOnModule(Module &M)
{
   ProfileInfo& PI = getAnalysis<ProfileInfo>();
   double AbsoluteTiming = 0.0, BlockTiming = 0.0, MpiTiming = 0.0, CallTiming = 0.0;
   double MpiTimingsize = 0.0;
   double MpiFittingTime = 0.0;
   double AllIrNum = 0.0;//add by haomeng. The num of ir
   double MPICallNUM = 0.0;//add by haomeng. The num of mpi callinst
   double AmountOfMpiComm = 0.0;//add by haomeng. The amount of commucation of mpi
   double RealMpiTime = 0.0;//add by haomeng. The real time of mpi
   double RealWaitTime = 0.0;//add by haomeng. The real wait time of mpi
   std::map<std::string, double> InstNum;
   std::map<std::string, double> InstTime;
   for(TimingSource* S : Sources){
      if (isa<BBlockTiming>(S)
          && BlockTiming < DBL_EPSILON) { // BlockTiming is Zero
         auto BT = cast<BBlockTiming>(S);
         for(Module::iterator F = M.begin(), FE = M.end(); F != FE; ++F){
            if(Ignore.count(F->getName())) continue;
#ifndef NDEBUG
            double FuncTiming = 0.;
            size_t MaxTimes = 0;
            double MaxCount = 0.;
            double MaxProd = 0.;
            StringRef MaxName;
            for(Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB){
               size_t exec_times = PI.getExecutionCount(BB);
               double exec_count = BT->count(*BB);
               double timing = exec_times * exec_count;
               if (isa<IrinstTiming>(BT))//add by haomeng.
               {
                  auto IRT = cast<IrinstTiming>(BT);
                  double IR_C = IRT->ir_count(*BB);
                  AllIrNum += (exec_times * IR_C);
                  for(BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; II++){
                     std::string strtmp = II->getOpcodeName();
                     try{
                        InstNum[strtmp]+=exec_times;
                     }
                     catch(std::out_of_range& e)
                     {
                        InstNum[strtmp] = exec_times;
                     }
                     try{
                        InstTime[strtmp]+=(exec_times*IRT->count(*II));
                     }
                     catch(std::out_of_range& e)
                     {
                        InstTime[strtmp] = (exec_times*IRT->count(*II));
                     }
                  }
               }


               if(timing > MaxProd){
                  MaxProd = timing;
                  MaxCount = exec_count;
                  MaxTimes = exec_times;
                  MaxName = BB->getName();
               }
               FuncTiming += timing; // 基本块频率×基本块时间
            }
            if (TimingDebug)
              outs() << FuncTiming << "\t"
                     << "max=" << MaxTimes << "*" << MaxCount << "\t" << MaxName
                     << "\t" << F->getName() << "\n";
            BlockTiming += FuncTiming;
#else
            for(Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB){
               BlockTiming += PI.getExecutionCount(BB) * S->count(*BB);
            }
#endif
         }
      }
      if(isa<MPITiming>(S) && MpiTiming < DBL_EPSILON){ // MpiTiming is Zero
         auto MT = cast<MPITiming>(S);
         auto S = PI.getAllTrapedValues(MPIFullInfo);
         auto U = PI.getAllTrapedValues(MPInfo);
         if(U.size()>0) outs()<<"Notice: Old Mpi Profiling Format\n";
         S.insert(S.end(), U.begin(), U.end());
//add by haomeng. Calculate the real time of mpi
         for(Module::iterator F = M.begin(), E = M.end(); F!= E; ++F){
            for(Function::iterator BB = F->begin(), BE = F->end(); BB!= BE; ++BB){
               for(BasicBlock::iterator I = BB->begin(), IE = BB->end(); I!= IE; ++I){
                  CallInst* CI = dyn_cast<CallInst>(&*I);
                  if(CI == NULL) continue;
                  Value* CV = const_cast<CallInst*>(CI)->getCalledValue();
                  Function* func = dyn_cast<Function>(lle::castoff(CV));
                  if(func == NULL)
                     errs()<<"No func!\n";
                  StringRef str = func->getName();
                  if(str.startswith("mpi_")){
                     if(str.startswith("mpi_init_")||str.startswith("mpi_comm_rank_")||str.startswith("mpi_comm_size_"))
                        continue;
                    RealMpiTime += PI.getMPITime(CI);
                    if(str.startswith("mpi_wait_")||str.startswith("mpi_barrier_")||str.startswith("mpi_waitall_"))
                       RealWaitTime += PI.getMPITime(CI);
                  }
               }
            }
         }


         for(auto I : S){
            const CallInst* CI = cast<CallInst>(I);
            const BasicBlock* BB = CI->getParent();
            if(Ignore.count(BB->getParent()->getName())) continue;

            //0 means num of processes fixed, 1 means datasize fixed
            double timing = MT->count(*I, PI.getExecutionCount(BB), PI.getExecutionCount(CI)); // IO 模型
            //double timingsize = MT->newcount(*I,PI.getExecutionCount(BB),PI.getExecutionCount(CI),1);
            double fittingtime = MT->fittingcount(*I,PI.getExecutionCount(BB),PI.getExecutionCount(CI));

            if(isa<LatencyTiming>(MT))//add by haomeng.
            {
               auto LTR = cast<LatencyTiming>(MT);
               size_t BFreq = PI.getExecutionCount(BB);
               MPICallNUM += BFreq;
               AmountOfMpiComm += LTR->Comm_amount(*I,BFreq,PI.getExecutionCount(CI));
            }

#ifdef NDEBUG
            if(TimingDebug)
               outs() << "  " << PI.getTrapedIndex(I)
                      << "\tBB:" << PI.getExecutionCount(BB) << "\tT:" << timing
                      << "N:" << BB->getParent()->getName() << ":"
                      << BB->getName() << "\n";
#endif
            MpiTiming += timing;
            //MpiTimingsize += timingsize*1000.0;
            MpiFittingTime += fittingtime;
         }
      }
      if(isa<LibCallTiming>(S) && CallTiming < DBL_EPSILON){
         auto CT = cast<LibCallTiming>(S);
         for(auto& F : M){
            for(auto& BB : F){
               for(auto& I : BB){
                  if(CallInst* CI = dyn_cast<CallInst>(&I)){
                     CallTiming += CT->count(*CI, PI.getExecutionCount(&BB));
                  }
               }
            }
         }
      }
   }
   AbsoluteTiming = BlockTiming + MpiTiming/*MpiTiming */+ CallTiming;
   outs()<<"Block Timing: "<<BlockTiming<<" ns\n";
   outs()<<"MPI Timing: "<<MpiTiming<<" ns\n";
   outs()<<"Call Timing: "<<CallTiming<<" ns\n";
   outs()<<"Timing: "<<AbsoluteTiming<<" ns\n";
   outs()<<"Inst Num: "<< AllIrNum << "\n";
   outs()<<"Mpi Num: "<< MPICallNUM<< "\n";
   outs()<<"Comm Amount: "<< AmountOfMpiComm<< "\n";
   outs()<<"Real MPI Timing: "<< RealMpiTime*pow(10,9) << " ns\n";
   outs()<<"Real MPI Wait Timing: "<< RealWaitTime*pow(10,9) << " ns\n";
   outs()<<"MPI Fitting Timing: "<< MpiFittingTime*pow(10,9) << " ns\n";
//   {
//      outs()<<"================The counts of instructions===========\n";
//      typedef std::pair<std::string, double> PAIR;
//      double allinstnum = 0.0;
//      std::vector<PAIR> maptovec(InstNum.begin(),InstNum.end());
//      std::sort(maptovec.begin(),maptovec.end(),[](PAIR& x, PAIR& y){return x.second > y.second;});
//      for(std::vector<PAIR>::iterator iter = maptovec.begin();iter != maptovec.end();iter++)
//      {
//         outs()<<iter->first<<"\t\t\t"<<iter->second<<"\n";
//         allinstnum += iter->second;
//      }
//      outs() << "All Num:\t"<<allinstnum<<"\n";
//
//      std::function<double(std::string)> JudgeGroup = [&InstNum](std::string opstr)->double
//      {
//
//         std::map<std::string,double>::iterator iter = InstNum.find(opstr);
//         if(iter == InstNum.end())
//            return 0;
//         else
//            return InstNum[opstr];
//      };
//      std::map<std::string, double> groupOps1 = 
//      {
//         {"load", JudgeGroup("load")}
//      };
//
//      std::map<std::string, double> groupOps2 = 
//      {
//         {"add", JudgeGroup("add")},
//         {"sub", JudgeGroup("sub")},
//         {"shl", JudgeGroup("shl")},
//         {"lshr", JudgeGroup("lshr")},
//         {"ashr", JudgeGroup("ashr")},
//         {"and", JudgeGroup("and")},
//         {"or", JudgeGroup("or")},
//         {"xor", JudgeGroup("xor")},
//         {"alloca", JudgeGroup("alloca")},
//         {"ptrtoint", JudgeGroup("ptrtoint")},
//         {"bitcast", JudgeGroup("bitcast")}
//      };
//
//      std::map<std::string, double> groupOps3 = 
//      {
//         {"fadd", JudgeGroup("fadd")},
//         {"fmul", JudgeGroup("fmul")},
//         {"fsub", JudgeGroup("fsub")},
//         {"mul", JudgeGroup("mul")},
//         {"select", JudgeGroup("select")},
//         {"fcmp", JudgeGroup("fcmp")},
//         {"icmp", JudgeGroup("icmp")},
//         {"inttoptr", JudgeGroup("inttoptr")},
//         {"zext", JudgeGroup("zext")},
//         {"sext", JudgeGroup("sext")},
//         {"trunc", JudgeGroup("trunc")},
//         {"uitofp", JudgeGroup("uitofp")}
//      };
//
//      std::map<std::string, double> groupOps4 = 
//      {
//         {"fptrunc", JudgeGroup("fptrunc")},
//         {"fpext", JudgeGroup("fpext")},
//         {"fptoui", JudgeGroup("fptoui")},
//         {"fptosi", JudgeGroup("fptosi")},
//         {"sitofp", JudgeGroup("sitofp")}
//      };
//
//      std::map<std::string, double> groupOps5 = 
//      {
//         {"udiv", JudgeGroup("udiv")},
//         {"sdiv", JudgeGroup("sdiv")},
//         {"fdiv", JudgeGroup("fdiv")},
//         {"urem", JudgeGroup("urem")},
//         {"srem", JudgeGroup("srem")},
//         {"frem", JudgeGroup("frem")}
//      };
//
//      std::map<std::string, double> groupOps6 = 
//      {
//         {"call", JudgeGroup("call")},
//         {"phi", JudgeGroup("phi")},
//         {"br", JudgeGroup("br")},
//         {"unreachable", JudgeGroup("unreachable")},
//         {"ret", JudgeGroup("ret")}
//      };
//      std::map<std::string, double> groupOps7 = 
//      {
//         {"store", JudgeGroup("store")}
//      };
//      std::map<std::string, double> groupOps8 = 
//      {
//         {"getelementptr", JudgeGroup("getelementptr")}
//      };
//      std::function<double(std::map<std::string, double>&) > GroupSum = [](std::map<std::string, double>& groupOps)->double
//      {
//
//         std::map<std::string,double>::iterator iter;
//         double sum = 0.0;
//         for(iter = groupOps.begin(); iter != groupOps.end(); iter++)
//            sum+=iter->second;
//
//         return sum;
//      };
//
//
//      outs() << "===============The ins count of Instruction Groups===========\n";
//      double groupall[8];
//      groupall[0] = GroupSum(groupOps1);
//      groupall[1] = GroupSum(groupOps2);
//      groupall[2] = GroupSum(groupOps3);
//      groupall[3] = GroupSum(groupOps4);
//      groupall[4] = GroupSum(groupOps5);
//      groupall[5] = GroupSum(groupOps6);
//      groupall[6] = GroupSum(groupOps7);
//      groupall[7] = GroupSum(groupOps8);
//      double allInstNum = 0.0;
//      for(int i = 0; i < 8;i++){
//         allInstNum+=groupall[i];
//         outs()<<groupall[i]<<" ";
//      }
//     // outs()<<"Group1:\t"<<groupall[0]<<"\n";
//     // outs()<<"Group2:\t"<<groupall[1]<<"\n";
//     // outs()<<"Group3:\t"<<groupall[2]<<"\n";
//     // outs()<<"Group4:\t"<<groupall[3]<<"\n";
//     // outs()<<"Group5:\t"<<groupall[4]<<"\n";
//     // outs()<<"Group6:\t"<<groupall[5]<<"\n";
//     // outs()<<"Group7:\t"<<groupall[6]<<"\n";
//     // outs()<<"Group8:\t"<<groupall[7]<<"\n";
//     // outs()<<"\nGroupAll:\t"<<allInstNum<<"\n";
//      outs()<<"\n======\n";
//      //double paras[8] = {-96.8128, -88.0085, -20.0678, 849.458, -2008.82, -83.3543, 189.144, 187.222};//ep,sp,lu:pred
//      double paras[8] = {49.5664, 66.8576, -2.734, -129.921, 333.339, -10.6375, -10.8649, -99.7951};//ft,mg,cg:pred
//      double predcomtime = 0.0;
//      for(int i = 0; i < 8;i++)
//         predcomtime+=(groupall[i]*paras[i]);
//      outs()<<"Pred computation time: "<<predcomtime+CallTiming <<"\n";
//   }
//   {
//      outs()<<"================The time of instructions===========\n";
//      typedef std::pair<std::string, double> PAIR;
//      double allinsttime = 0.0;
//      std::vector<PAIR> maptovec(InstTime.begin(),InstTime.end());
//      std::sort(maptovec.begin(),maptovec.end(),[](PAIR& x, PAIR& y){return x.second > y.second;});
//      for(std::vector<PAIR>::iterator iter = maptovec.begin();iter != maptovec.end();iter++)
//      {
//         outs()<<iter->first<<"\t\t\t"<<iter->second<<"\n";
//         allinsttime += iter->second;
//      }
//      outs() << "All Num:\t"<<allinsttime<<"\n";
//
//      std::function<double(std::string)> JudgeGroup = [&InstTime](std::string opstr)->double
//      {
//
//         std::map<std::string,double>::iterator iter = InstTime.find(opstr);
//         if(iter == InstTime.end())
//            return 0;
//         else
//            return InstTime[opstr];
//      };
//      std::map<std::string, double> groupOps1 = 
//      {
//         {"load", JudgeGroup("load")},
//         {"store", JudgeGroup("store")},
//         {"alloca", JudgeGroup("alloca")},
//         {"getelementptr", JudgeGroup("getelementptr")}
//
//      };
//
//      std::map<std::string, double> groupOps2 = 
//      {
//         {"fadd", JudgeGroup("fadd")},
//         {"fmul", JudgeGroup("fmul")},
//         {"fsub", JudgeGroup("fsub")},
//         {"fcmp", JudgeGroup("fcmp")},
//         {"uitofp", JudgeGroup("uitofp")},
//         {"fptrunc", JudgeGroup("fptrunc")},
//         {"fpext", JudgeGroup("fpext")},
//         {"fptoui", JudgeGroup("fptoui")},
//         {"fptosi", JudgeGroup("fptosi")},
//         {"sitofp", JudgeGroup("sitofp")},
//         {"fdiv", JudgeGroup("fdiv")},
//         {"frem", JudgeGroup("frem")}
//
//      };
//      std::map<std::string, double> groupOps3 = 
//      {
//         {"add", JudgeGroup("add")},
//         {"sub", JudgeGroup("sub")},
//         {"shl", JudgeGroup("shl")},
//         {"lshr", JudgeGroup("lshr")},
//         {"ashr", JudgeGroup("ashr")},
//         {"and", JudgeGroup("and")},
//         {"or", JudgeGroup("or")},
//         {"xor", JudgeGroup("xor")},
//         {"ptrtoint", JudgeGroup("ptrtoint")},
//         {"bitcast", JudgeGroup("bitcast")},
//         {"mul", JudgeGroup("mul")},
//         {"select", JudgeGroup("select")},
//         {"icmp", JudgeGroup("icmp")},
//         {"inttoptr", JudgeGroup("inttoptr")},
//         {"zext", JudgeGroup("zext")},
//         {"sext", JudgeGroup("sext")},
//         {"trunc", JudgeGroup("trunc")},
//         {"udiv", JudgeGroup("udiv")},
//         {"sdiv", JudgeGroup("sdiv")},
//         {"urem", JudgeGroup("urem")},
//         {"srem", JudgeGroup("srem")}
//      };
//
//      std::function<double(std::map<std::string, double>&) > GroupSum = [](std::map<std::string, double>& groupOps)->double
//      {
//
//         std::map<std::string,double>::iterator iter;
//         double sum = 0.0;
//         for(iter = groupOps.begin(); iter != groupOps.end(); iter++)
//            sum+=iter->second;
//
//         return sum;
//      };
//
//
//      outs() << "===============The ins time of Instruction Groups===========\n";
//      double groupall[3];
//      groupall[0] = GroupSum(groupOps1);
//      groupall[1] = GroupSum(groupOps2);
//      groupall[2] = GroupSum(groupOps3);
//      double allInsttime = 0.0;
//      for(int i = 0; i < 3;i++){
//         allInsttime+=groupall[i];
//         outs()<<groupall[i]<<" ";
//      }
//      outs()<<"\n";
//      outs()<<"Group1:\t"<<groupall[0]<<"\n";
//      outs()<<"Group2:\t"<<groupall[1]<<"\n";
//      outs()<<"Group3:\t"<<groupall[2]<<"\n";
//      outs()<<"\nGroupAll:\t"<<allInsttime<<"\n";
//      outs()<<"\n======\n";
//      //double paras[8] = {-96.8128, -88.0085, -20.0678, 849.458, -2008.82, -83.3543, 189.144, 187.222};//ep,sp,lu:pred
//      double paras[3] = {0.1, 0.2,0.7};//mpi,bt,pred
//      //double paras[3] = {0.5, 0.2,0.7};//mpi,ft,pred
//      //double paras[3] = {0.15, 0.2,0.7};//mpi,lu,pred
//      double predcomtime = 0.0;
//      for(int i = 0; i < 3;i++)
//         predcomtime+=(groupall[i]*paras[i]);
//      outs()<<"Pred computation time1: "<<predcomtime+CallTiming <<"\n";
//   }

   return false;
}

ProfileTimingPrint::ProfileTimingPrint(std::vector<TimingSource*>&& TS,
      std::vector<std::string>& Files):ModulePass(ID), Sources(TS)
{
   if(Sources.size() > Files.size()){
      errs()<<"No Enough File to initialize Timing Source\n";
      exit(-1);
   }
   if(TimingIgnore!=""){
      std::ifstream IgnoreFile(TimingIgnore);
      if(!IgnoreFile.is_open()){
         errs()<<"Couldn't open ignore file: "<<TimingIgnore<<"\n";
         exit(-1);
      }
      std::copy(std::istream_iterator<std::string>(IgnoreFile),
                std::istream_iterator<std::string>(),
                std::inserter(Ignore, Ignore.end()));
      IgnoreFile.close();
   }
   for(unsigned i = 0; i < Sources.size(); ++i){
      Sources[i]->init_with_file(Files[i].c_str());
#ifndef NDEBUG
      if(TimingDebug){
         outs()<<"parsed "<<Files[i]<<" file's content:\n";
         Sources[i]->print(outs());
      }
#endif
   }
}

ProfileTimingPrint::~ProfileTimingPrint()
{
   for(auto S : Sources)
      delete S;
}
