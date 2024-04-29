def _grad_step(
        maml,
        task,
        opt,
        schedule
) -> None:
    "Subroutine to perform gradient step"

    # normalize loss with the number of task
    for _, p in maml.named_parameters():
        p.grad.data.mul_(1.0 / (task + 1))

    opt.step()
    schedule.step()