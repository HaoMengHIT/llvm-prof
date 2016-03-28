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
   outs()<<"Hello world\n";
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
         outs()<<str<<"\t"<<(double)(MpiComm/BFreq)<<"\t" << MpiComm<<"\t"<<BFreq<<"\n";
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
   double AllIrNum = 0.0;//add by haomeng. The num of ir
   double MPICallNUM = 0.0;//add by haomeng. The num of mpi callinst
   double AmountOfMpiComm = 0.0;//add by haomeng. The amount of commucation of mpi
   double RealMpiTime = 0.0;//add by haomeng. The real time of mpi
   double RealWaitTime = 0.0;//add by haomeng. The real wait time of mpi
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
            double timingsize = MT->newcount(*I,PI.getExecutionCount(BB),PI.getExecutionCount(CI),1);

            if(isa<LatencyTiming>(MT))//add by haomeng.
            {
               auto LTR = cast<LatencyTiming>(MT);
               size_t BFreq = PI.getExecutionCount(BB);
               MPICallNUM += BFreq;
               AmountOfMpiComm += PI.getExecutionCount(CI);//LTR->Comm_amount(*I,BFreq,PI.getExecutionCount(CI));
            }

#ifdef NDEBUG
            if(TimingDebug)
               outs() << "  " << PI.getTrapedIndex(I)
                      << "\tBB:" << PI.getExecutionCount(BB) << "\tT:" << timing
                      << "N:" << BB->getParent()->getName() << ":"
                      << BB->getName() << "\n";
#endif
            MpiTiming += timing;
            MpiTimingsize += timingsize*1000.0;
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
   AbsoluteTiming = BlockTiming + MpiTimingsize/*MpiTiming */+ CallTiming;
   outs()<<"Block Timing: "<<BlockTiming<<" ns\n";
   outs()<<"MPI Timing1: "<<MpiTimingsize<<" ns\n";
   outs()<<"Call Timing: "<<CallTiming<<" ns\n";
   outs()<<"Timing: "<<AbsoluteTiming<<" ns\n";
   outs()<<"MPI Timing: "<<MpiTiming<<" ns\n";
   outs()<<"Inst Num: "<< AllIrNum << "\n";
   outs()<<"Mpi Num: "<< MPICallNUM<< "\n";
   outs()<<"Comm Amount: "<< AmountOfMpiComm<< "\n";
   outs()<<"Real MPI Timing: "<< RealMpiTime*pow(10,9) << " ns\n";
   outs()<<"Real MPI Wait Timing: "<< RealWaitTime*pow(10,9) << " ns\n";
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
