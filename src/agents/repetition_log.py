"""
Logging infrastructure for repetition detection decisions.

This module provides comprehensive logging and analysis tools for
tracking similarity detection decisions, enabling human review and
system improvement.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Single log entry for a similarity decision"""
    timestamp: str
    action: str
    reason: str
    similarity_score: float
    metadata: Dict[str, Any]
    text_preview: str
    full_decision: Dict[str, Any]
    tier: str
    agent_id: Optional[str] = None
    iteration: Optional[int] = None
    revision_number: Optional[int] = None


class RepetitionLog:
    """
    Stores and analyzes repetition detection decisions.
    
    Provides:
    1. Human review queue for flagged content
    2. System improvement metrics
    3. Debugging information for skipped content
    4. Statistical analysis of decisions
    """
    
    def __init__(self, session_id: str, log_dir: Optional[str] = None):
        """
        Initialize the repetition logger.
        
        Args:
            session_id: Unique session identifier
            log_dir: Directory for log files (defaults to outputs/similarity_logs)
        """
        self.session_id = session_id
        self.session_log: List[LogEntry] = []
        self.review_queue: List[LogEntry] = []
        self.skip_log: List[LogEntry] = []
        
        # Set up log directory
        if log_dir is None:
            log_dir = Path("outputs") / "similarity_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"similarity_log_{session_id}_{timestamp}.json"
        self.review_file = self.log_dir / f"review_queue_{session_id}_{timestamp}.json"
        self.stats_file = self.log_dir / f"stats_{session_id}_{timestamp}.json"
        
        logger.info(f"RepetitionLog initialized for session {session_id}")
        logger.info(f"Log file: {self.log_file}")
    
    def log_decision(
        self,
        decision: Dict[str, Any],
        original_text: str,
        metadata: Dict[str, Any]
    ):
        """
        Log a similarity detection decision.
        
        Args:
            decision: The similarity decision made
            original_text: The full text that was analyzed
            metadata: Additional metadata about the content
        """
        # Get action as string (handle enum or string)
        action = decision.get('action', 'UNKNOWN')
        if hasattr(action, 'value'):
            # It's an enum, get the value
            action = action.value
        elif hasattr(action, 'name'):
            # It's an enum, get the name
            action = action.name
        else:
            # Convert to string
            action = str(action)
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            reason=decision.get('reason', ''),
            similarity_score=decision.get('similarity_score', 0.0),
            metadata=metadata,
            text_preview=original_text[:500] + "..." if len(original_text) > 500 else original_text,
            full_decision=decision,
            tier=decision.get('tier', 'UNKNOWN'),
            agent_id=metadata.get('agent_id'),
            iteration=metadata.get('iteration'),
            revision_number=metadata.get('revision_number')
        )
        
        # Add to session log
        self.session_log.append(entry)
        
        # Add to appropriate queue based on action
        action_upper = action.upper() if isinstance(action, str) else str(action)
        
        if action_upper == 'FLAG':
            self.review_queue.append(entry)
            logger.info(f"Added to review queue: {entry.reason}")
        elif action_upper == 'SKIP':
            self.skip_log.append(entry)
            logger.info(f"Content skipped: {entry.reason}")
        
        # Save to file periodically (every 10 entries)
        if len(self.session_log) % 10 == 0:
            self._save_logs()
    
    def add_human_review(
        self,
        entry_index: int,
        human_decision: str,
        human_notes: str
    ):
        """
        Add human review to a flagged entry.
        
        Args:
            entry_index: Index in review queue
            human_decision: Human's decision (ACCEPT/REJECT/MODIFY)
            human_notes: Human's notes about the decision
        """
        if 0 <= entry_index < len(self.review_queue):
            entry = self.review_queue[entry_index]
            entry.metadata['human_review'] = {
                'decision': human_decision,
                'notes': human_notes,
                'reviewed_at': datetime.now().isoformat()
            }
            logger.info(f"Added human review for entry {entry_index}")
            self._save_logs()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the session.
        
        Returns:
            Dictionary with various statistics about decisions
        """
        total_checks = len(self.session_log)
        
        if total_checks == 0:
            return {
                'total_checks': 0,
                'message': 'No similarity checks performed yet'
            }
        
        # Count by action (normalize to uppercase strings)
        action_counts = {
            'skipped': sum(1 for e in self.session_log if str(e.action).upper() == 'SKIP'),
            'flagged': sum(1 for e in self.session_log if str(e.action).upper() == 'FLAG'),
            'accepted': sum(1 for e in self.session_log if str(e.action).upper() == 'ACCEPT')
        }
        
        # Count by tier
        tier_counts = {}
        for entry in self.session_log:
            tier = entry.tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Similarity score statistics
        scores = [e.similarity_score for e in self.session_log if e.similarity_score > 0]
        
        score_stats = {}
        if scores:
            score_stats = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(min(scores)),
                'max': float(max(scores)),
                'percentile_25': float(np.percentile(scores, 25)),
                'percentile_75': float(np.percentile(scores, 75))
            }
        
        # Action percentages
        action_percentages = {
            f'{action}_percentage': (count / total_checks * 100)
            for action, count in action_counts.items()
        }
        
        # Review queue status
        review_status = {
            'pending_review': len([e for e in self.review_queue 
                                  if 'human_review' not in e.metadata]),
            'reviewed': len([e for e in self.review_queue 
                           if 'human_review' in e.metadata])
        }
        
        # Most common skip reasons
        skip_reasons = {}
        for entry in self.skip_log:
            reason = entry.reason
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        
        # Sort skip reasons by frequency
        top_skip_reasons = sorted(
            skip_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        statistics = {
            'total_checks': total_checks,
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'tier_distribution': tier_counts,
            'similarity_scores': score_stats,
            'review_status': review_status,
            'top_skip_reasons': top_skip_reasons,
            'session_id': self.session_id,
            'generated_at': datetime.now().isoformat()
        }
        
        return statistics
    
    def export_review_queue(self) -> str:
        """
        Export flagged content for human review.
        
        Returns:
            JSON string of review queue
        """
        review_data = []
        
        for idx, entry in enumerate(self.review_queue):
            review_item = {
                'index': idx,
                'timestamp': entry.timestamp,
                'reason': entry.reason,
                'similarity_score': entry.similarity_score,
                'tier': entry.tier,
                'text_preview': entry.text_preview,
                'metadata': entry.metadata,
                'recommendation': entry.full_decision.get('recommendation', ''),
                'analysis': entry.full_decision.get('analysis', {}),
                'human_review': entry.metadata.get('human_review')
            }
            review_data.append(review_item)
        
        return json.dumps(review_data, indent=2)
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of the session.
        
        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        
        report_lines = [
            "=" * 70,
            f"SIMILARITY DETECTION REPORT",
            f"Session: {self.session_id}",
            f"Generated: {stats['generated_at']}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Checks: {stats['total_checks']}",
            ""
        ]
        
        # Action breakdown
        if stats['total_checks'] > 0:
            report_lines.extend([
                "DECISIONS",
                "-" * 40
            ])
            
            for action, count in stats['action_counts'].items():
                percentage = stats['action_percentages'][f'{action}_percentage']
                report_lines.append(f"  {action.title():10} {count:5} ({percentage:5.1f}%)")
            
            report_lines.append("")
            
            # Tier distribution
            if stats['tier_distribution']:
                report_lines.extend([
                    "TIER DISTRIBUTION",
                    "-" * 40
                ])
                
                for tier, count in sorted(stats['tier_distribution'].items()):
                    percentage = (count / stats['total_checks']) * 100
                    report_lines.append(f"  {tier:15} {count:5} ({percentage:5.1f}%)")
                
                report_lines.append("")
            
            # Similarity scores
            if stats['similarity_scores']:
                scores = stats['similarity_scores']
                report_lines.extend([
                    "SIMILARITY SCORES",
                    "-" * 40,
                    f"  Mean:     {scores['mean']:.1%}",
                    f"  Median:   {scores['median']:.1%}",
                    f"  Std Dev:  {scores['std']:.1%}",
                    f"  Range:    {scores['min']:.1%} - {scores['max']:.1%}",
                    ""
                ])
            
            # Review queue
            report_lines.extend([
                "REVIEW QUEUE",
                "-" * 40,
                f"  Pending Review: {stats['review_status']['pending_review']}",
                f"  Reviewed:       {stats['review_status']['reviewed']}",
                ""
            ])
            
            # Top skip reasons
            if stats['top_skip_reasons']:
                report_lines.extend([
                    "TOP SKIP REASONS",
                    "-" * 40
                ])
                
                for reason, count in stats['top_skip_reasons']:
                    report_lines.append(f"  {count:3}x {reason[:60]}")
                
                report_lines.append("")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def _save_logs(self):
        """Save logs to files."""
        try:
            # Save full session log
            with open(self.log_file, 'w') as f:
                log_data = [asdict(entry) for entry in self.session_log]
                json.dump(log_data, f, indent=2)
            
            # Save review queue
            with open(self.review_file, 'w') as f:
                review_data = [asdict(entry) for entry in self.review_queue]
                json.dump(review_data, f, indent=2)
            
            # Save statistics
            with open(self.stats_file, 'w') as f:
                json.dump(self.get_statistics(), f, indent=2)
            
            logger.debug(f"Logs saved to {self.log_dir}")
            
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
    
    def close(self):
        """Close the logger and save final logs."""
        self._save_logs()
        
        # Generate and save final report
        report = self.generate_report()
        report_file = self.log_dir / f"report_{self.session_id}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Final report saved to {report_file}")
        logger.info(f"Review queue: {len(self.review_queue)} items")
        logger.info(f"Skip log: {len(self.skip_log)} items")