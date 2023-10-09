from typing import List

from signal_slot.signal_slot import EventLoop, EventLoopProcess

from sample_factory.algo.learning.learner_worker import init_learner_process
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.sampler import ParallelSampler
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.utils.typing import StatusCode
from sample_factory.utils.utils import log


class ParallelRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.processes: List[EventLoopProcess] = []

    def init(self) -> StatusCode:
        status = super().init()
        if status != ExperimentStatus.SUCCESS:
            return status

        mp_ctx = get_mp_ctx(self.cfg.serial_mode)

        for policy_id in range(self.cfg.num_policies):
            self.batchers[policy_id] = []
            for env_info, buffer_mgr in zip(self.env_info, self.buffers_mgr):
                batcher_event_loop = EventLoop(f"batcher_evt_loop_{env_info.name}")
                batcher = self._make_batcher(batcher_event_loop, policy_id, buffer_mgr, env_info)
                self.batchers[policy_id].append(batcher)
                batcher_event_loop.owner = batcher

            learner_proc = EventLoopProcess(f"learner_proc{policy_id}", mp_ctx, init_func=init_learner_process)
            self.processes.append(learner_proc)

            self.learners[policy_id] = self._make_learner(
                learner_proc.event_loop,
                policy_id,
                self.batchers[policy_id],
            )
            learner_proc.event_loop.owner = self.learners[policy_id]
            learner_proc.set_init_func_args((sf_global_context(), self.learners[policy_id]))

        self.samplers = []
        for env_info, buffer_mgr in zip(self.env_info, self.buffers_mgr):
            sampler = self._make_sampler(ParallelSampler, self.event_loop, buffer_mgr, env_info)
            self.samplers.append(sampler)

        self.connect_components()
        return status

    def _on_start(self):
        self._start_processes()
        super()._on_start()

    def _start_processes(self):
        log.debug("Starting all processes...")
        for p in self.processes:
            log.debug(f"Starting process {p.name}")
            p.start()
            self.event_loop.process_events()

    def _on_everything_stopped(self):
        for p in self.processes:
            log.debug(f"Waiting for process {p.name} to stop...")
            p.join()

        for sampler in self.samplers:
            sampler.join()
        super()._on_everything_stopped()
