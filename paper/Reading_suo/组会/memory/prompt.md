### prompt



Fine-tuning:

• 预训练模型“迁就“各种下游任务。通过引入各种辅助任务loss，在下游任务上fine-tuning预训练模型以便让其更加适配下游任务。因此，预训练模型做出了更多“牺牲“ 。

• Prompting:

 • 各种下游任务“迁就“预训练语言模型。需要对不同下游 任务进行重构，使得下游任务贴近模型在预训练过程中解 决的任务。这个过程中，下游任务做出了更多“牺牲“ 。



Alec Radford et al, Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.

